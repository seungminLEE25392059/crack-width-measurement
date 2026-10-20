"""Lingyu Wu: lossless-cache replay, micro aggregation and coverage guard tests."""
import json,tempfile,unittest
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src import method_c as c,method_d as d
from run_method_cd_validation import select_threshold,array_sha,checked_cache
from run_method_d_width_validation import aggregate_image_rows,SUM_KEYS,WIDTH_KEYS
from run_method_cd_team_validation import ProbabilityReplay,entry

class ValidationPipelineTests(unittest.TestCase):
    def test_successful_cli_system_exit_does_not_skip_following_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved=sys.argv[:]
            with patch('run_method_cd_team_validation.runpy.run_path',side_effect=SystemExit(0)) as run:
                entry('check_interfaces.py',[],Path(tmp)/'interface.log')
                entry('run_evaluation.py',['--split','val'],Path(tmp)/'evaluation.log')
                self.assertEqual(run.call_count,2)
            self.assertEqual(sys.argv,saved)
    def test_failed_cli_exit_stops_pipeline_and_restores_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved=sys.argv[:]
            with patch('run_method_cd_team_validation.runpy.run_path',side_effect=SystemExit(2)):
                with self.assertRaises(SystemExit) as caught:
                    entry('check_interfaces.py',[],Path(tmp)/'failed.log')
                self.assertEqual(caught.exception.code,2)
            self.assertEqual(sys.argv,saved)
    def row(self,name,tp,fp,fn,matched,mae):
        row={'variant':name,'tp':tp,'fp':fp,'fn':fn,'matched_width_samples':matched,
             'postprocessing_seconds':.25}
        row.update({key:0 for key in SUM_KEYS if key!='matched_width_samples'})
        row.update({key:mae for key in WIDTH_KEYS})
        return row
    def test_threshold_objective_and_tie_breaks(self):
        rows=[{'threshold':.4,'f1':.8,'recall':.9},{'threshold':.5,'f1':.8,'recall':.9},
              {'threshold':.9,'f1':.79,'recall':.99}]
        self.assertEqual(select_threshold(rows)['threshold'],.5)
        rows[0]['recall']=.91;self.assertEqual(select_threshold(rows)['threshold'],.4)
    def test_aggregate_is_micro_and_widths_weight_matched_references(self):
        a=self.row('A',1,0,0,1,10.);b=self.row('A',9,9,0,3,2.)
        result=aggregate_image_rows([a,b],['A'])[0]
        self.assertAlmostEqual(result['f1'],20/29)
        self.assertAlmostEqual(result['mean_width_mae'],4.)
        self.assertEqual(result['matched_width_samples'],4);self.assertEqual(result['validation_images'],2)
    def test_no_matches_produces_null_errors(self):
        row=self.row('A',0,0,20,0,None);row['detection_failures']=2
        result=aggregate_image_rows([row],['A'])[0]
        self.assertIsNone(result['mean_width_mae']);self.assertEqual(result['detection_failures'],2)
    def test_original_width_guards_reject_better_mae_if_coverage_worse(self):
        baseline={'variant':'Baseline','recall':.9,'detection_failures':1,'max_width_bias':2.,'max_width_mae':4.,'mean_width_mae':2.}
        good={**baseline,'variant':'good','max_width_bias':.2,'max_width_mae':3.,'mean_width_mae':1.}
        bad={**good,'variant':'bad','detection_failures':2,'max_width_mae':.1}
        self.assertEqual(d.select_width_candidate([baseline,bad,good])['variant'],'good')
        bad={**good,'recall':.8};self.assertIsNone(d.select_width_candidate([baseline,bad]))
        bad={**good,'max_width_bias':-2};self.assertIsNone(d.select_width_candidate([baseline,bad]))
    def cache_fixture(self,root):
        image=np.zeros((3,4,3),np.uint8)
        prob=np.array([[0,.49,.5,1],[.1,.7,.2,.6],[0,0,0,0]],np.float32)
        path=root/'probabilities';path.mkdir();target=path/'v_prob.npy';np.save(target,prob)
        manifest={'images':{'v':{'sha256':c.file_sha256(target),'shape':[3,4],'decoded_image_sha256':array_sha(image)}}}
        protocol={'inference_params':{'depth':4,'base_channels':32,'tile':256,'overlap':32,'batch_size':8,'normalization':'RGB uint8 / 255'}}
        return image,prob,manifest,protocol
    def test_replay_preserves_exact_float_probabilities_and_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image,prob,manifest,protocol=self.cache_fixture(root)
            replay=ProbabilityReplay(root,protocol,manifest)
            net=SimpleNamespace(checkpoint_metadata={'epoch':19})
            mask,loaded=replay(image,net,{**protocol['inference_params'],'threshold':.5})
            np.testing.assert_array_equal(loaded,prob);np.testing.assert_array_equal(mask,prob>=.5)
            self.assertEqual(mask.dtype,np.bool_);self.assertEqual(replay.seen,['v'])
    def test_replay_rejects_wrong_image_epoch_and_inference_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image,prob,manifest,protocol=self.cache_fixture(root)
            replay=ProbabilityReplay(root,protocol,manifest);p={**protocol['inference_params'],'threshold':.5}
            with self.assertRaisesRegex(ValueError,'epoch'):replay(image,SimpleNamespace(checkpoint_metadata={'epoch':29}),p)
            net=SimpleNamespace(checkpoint_metadata={'epoch':19})
            with self.assertRaisesRegex(ValueError,'setting'):replay(image,net,{**p,'overlap':0})
            with self.assertRaisesRegex(ValueError,'Image'):replay(image+1,net,p)
    def test_lossless_cache_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,manifest,_=self.cache_fixture(root)
            target=root/'probabilities/v_prob.npy';target.write_bytes(target.read_bytes()+b'edited')
            with self.assertRaisesRegex(ValueError,'SHA'):checked_cache(root,'v',manifest)

if __name__=='__main__':unittest.main(verbosity=2)
