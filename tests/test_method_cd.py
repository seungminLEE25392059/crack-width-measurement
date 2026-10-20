"""Independent C/D correctness tests. Run: python -B -m unittest discover -s tests -p test_method_cd.py -v.
Owner: Lingyu Wu. Synthetic fixtures test mechanics, never report performance.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from src import config, data, method_c as c, method_d as d

torch.set_num_threads(4)
P = {**c.DEFAULT_PARAMS, "depth":3,"base_channels":16,"batch_size":2,"tile":64,"overlap":32,"max_epochs":2}


class MethodCTest(unittest.TestCase):
    def test_forward_backward_update_and_seed(self):
        c.seed_training(42)
        net = c.build_unet(P)
        initial = copy.deepcopy(net.state_dict())
        c.seed_training(42)
        second = c.build_unet(P)
        self.assertTrue(all(torch.equal(v,second.state_dict()[k]) for k,v in initial.items()))
        x = torch.rand(2,3,64,64)
        target = (torch.rand(2,1,64,64)>.98).float()
        valid = torch.ones_like(target);valid[:,:,-5:]=0
        optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
        logits = net(x)
        self.assertEqual(tuple(logits.shape),(2,1,64,64))
        loss = c.weighted_bce_dice(logits,target,valid,40)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in net.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(initial['head.weight'],net.state_dict()['head.weight']))

    def test_default_four_level_network(self):
        net=c.build_unet(c.DEFAULT_PARAMS).eval()
        with torch.inference_mode(): output=net(torch.zeros(1,3,256,256))
        self.assertEqual(tuple(output.shape),(1,1,256,256))
        self.assertEqual(len(net.encoders),4)

    def test_padding_and_empty_tile_loss(self):
        logits=torch.randn(2,1,8,8,requires_grad=True)
        valid=torch.zeros_like(logits);valid[:,:,:4,:5]=1
        for target in [torch.zeros_like(logits),torch.ones_like(logits)]:
            first=c.weighted_bce_dice(logits,target,valid,50)
            alternate=logits.detach().clone();alternate[valid==0]=1000
            self.assertAlmostEqual(float(first.detach()),float(c.weighted_bce_dice(alternate,target,valid,50)),places=6)
            first.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(first))
            self.assertTrue((logits.grad[valid==0]==0).all())
        self.assertEqual(float(c.weighted_bce_dice(logits,torch.zeros_like(logits),valid*0,50).detach()),0)

    def test_full_image_probability_fusion(self):
        class PixelOracle(nn.Module):
            def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.tensor(0.))
            def forward(self,x):return torch.logit(x[:,:1].clamp(.001,.999))+self.anchor*0
        image=np.random.default_rng(2).integers(0,256,(257,319,3),dtype=np.uint8)
        net=PixelOracle().train()
        mask,prob=c.segment_c(image,net,P)
        expected=np.clip(image[...,0]/255,.001,.999)
        np.testing.assert_allclose(prob,expected,atol=2e-7)
        np.testing.assert_array_equal(mask,prob>=.5)
        self.assertEqual(prob.dtype,np.float32)
        self.assertFalse(net.training)
        self.assertEqual(prob.shape,(257,319))
        small=c.segment_c(image[:7,:9],net,P)[1]
        np.testing.assert_allclose(small,expected[:7,:9],atol=2e-7)

    def test_checkpoint_round_trip(self):
        net=c.build_unet(P).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            torch.save({'format_version':1,'params':P,'model_state':net.state_dict(),'epoch':1,'val_dice':0.,'run_kind':'smoke'},path)
            restored=c.load_unet(path,'cpu')
            x=torch.randn(1,3,64,64)
            with torch.inference_mode():torch.testing.assert_close(net(x),restored(x),rtol=0,atol=0)
            self.assertFalse(restored.training)
            # Team interface checker supplies defaults even for CPU architecture.
            mask, prob = c.segment_c(np.zeros((27,31,3),np.uint8),restored,c.DEFAULT_PARAMS)
            self.assertEqual(mask.shape,(27,31))
            self.assertIsNone(c.load_unet(Path(tmp)/'missing.pt'))

    def test_dataset_geometry_padding_and_provenance(self):
        image=np.zeros((37,41,3),np.uint8);image[9:21,13:23]=255
        mask=image[...,0]>127
        with patch.object(data,'load_pair',return_value=(image,mask)):
            ds=c.TileDataset(['source'],P,True)
            c.seed_training(42);x,y,valid=ds[0]
            np.testing.assert_array_equal(x[0].numpy()>.5,y[0].numpy().astype(bool))
            self.assertEqual(int(valid.sum()),37*41)
            self.assertEqual(ds.items[0][:3],('source',0,0))
            self.assertEqual(ds.stats['foreground_pixels'],int(mask.sum()))

    def test_validation_ids_reject_test(self):
        with self.assertRaises(ValueError):c._check_ids(data.load_split()['test'][:1],'val')
        with self.assertRaises(ValueError):c._params({**P,'overlap':64})
        with self.assertRaises(ValueError):c.segment_c(np.zeros((3,3,3),np.uint8),None,P)

    def test_strict_metrics_match_team(self):
        from src import evaluate
        rng=np.random.default_rng(1)
        for a,b in [(rng.random((10,10))>.7,rng.random((10,10))>.8),(np.zeros((5,5),bool),np.zeros((5,5),bool))]:
            ours=c._scores(c._strict_counts(a,b));shared=evaluate.seg_metrics(a,b)
            for k in ['precision','recall','f1','iou']:self.assertEqual(ours[k],shared[k])


class MethodDTest(unittest.TestCase):
    def test_boundary_and_cross_independent_oracle(self):
        np.testing.assert_array_equal(d.EROSION_FOOTPRINT,[[0,1,0],[1,1,1],[0,1,0]])
        mask=np.random.default_rng(7).random((17,21))>.2
        padded=np.pad(mask,1,constant_values=True)
        expected=padded[1:-1,1:-1]&padded[:-2,1:-1]&padded[2:,1:-1]&padded[1:-1,:-2]&padded[1:-1,2:]
        np.testing.assert_array_equal(d.refine_d(mask,{'use_filter':False}),expected)
        full=np.ones((5,5),bool)
        self.assertTrue(d.refine_d(full,{'use_filter':False}).all())

    def test_empty_singleton_line_wide_branch_noise(self):
        fixtures=[np.zeros((20,20),bool),np.zeros((0,0),bool)]
        singleton=np.zeros((20,20),bool);singleton[10,10]=True;fixtures.append(singleton)
        line=np.zeros((20,20),bool);line[10,3:17]=True;fixtures.append(line)
        wide=np.zeros((20,20),bool);wide[5:15,4:16]=True;fixtures.append(wide)
        branch=np.zeros((20,20),bool);branch[3:17,9:12]=True;branch[8:11,2:18]=True;fixtures.append(branch)
        noise=wide.copy();noise[0,0]=noise[19,19]=True;fixtures.append(noise)
        for mask in fixtures:
            original=mask.copy();result=d.refine_d(mask,{'min_area':0,'min_elongation':1})
            self.assertEqual(result.shape,mask.shape);self.assertEqual(result.dtype,bool)
            np.testing.assert_array_equal(mask,original);self.assertTrue(np.all(result<=mask))
        self.assertFalse(d.refine_d(line,{'use_filter':False}).any())
        self.assertFalse(d.refine_d(singleton,{'use_filter':False}).any())
        self.assertTrue(d.refine_d(wide,{'use_filter':False}).any())

    def test_area_axis_ratio_and_eight_connectivity(self):
        mask=np.zeros((30,30),bool);mask[1:5,1:5]=True;mask[15,3:23]=True;mask[28,28]=True
        result=d.filter_components_d(mask,10,3)
        self.assertEqual(int(result.sum()),20)
        diag=np.eye(5,dtype=bool)
        np.testing.assert_array_equal(d.filter_components_d(diag,5,2),diag)
        self.assertFalse(d.filter_components_d(diag,6,0).any())
        single=np.ones((1,1),bool)
        self.assertTrue(d.filter_components_d(single,1,1).any())
        self.assertFalse(d.filter_components_d(single,1,2).any())

    def test_operation_order_controls(self):
        mask=np.zeros((30,30),bool);mask[3:25,10:15]=True
        p={**d.DEFAULT_PARAMS,'min_area':10,'min_elongation':1}
        filtered=d.filter_components_d(mask,10,1)
        expected=d.refine_d(filtered,{**p,'use_filter':False})
        np.testing.assert_array_equal(d.refine_d(mask,p),expected)
        original=d.refine_d(mask,{**p,'use_filter':False,'use_erosion':False})
        np.testing.assert_array_equal(original,mask);self.assertFalse(np.shares_memory(original,mask))
        with self.assertRaises(ValueError):d.refine_d(mask,{'erosion_radius':2})

    def test_infeasible_search_not_final(self):
        baseline={'variant':'Baseline','recall':1.,'f1':1.,'detection_failures':0,'max_width_bias':None}
        candidate={**baseline,'variant':'area=0,ratio=1.0','recall':0.,'f1':0.,'detection_failures':1}
        with tempfile.TemporaryDirectory() as tmp, patch.object(d,'evaluate_variants',return_value=([baseline,candidate],[])):
            selected,_=d.tune_d(data.load_split()['val'][:1],Path(tmp),.5,Path(tmp),areas=(0,),elongations=(1.,))
            self.assertFalse(selected['selection_feasible']);self.assertFalse(selected['selection_complete'])


if __name__=='__main__':unittest.main(verbosity=2)
