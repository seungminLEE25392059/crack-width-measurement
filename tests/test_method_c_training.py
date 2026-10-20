"""C/D training infrastructure tests. Synthetic fixtures test recovery, not performance.
Owner: Lingyu Wu. Run with python -B -m unittest discover -s tests -p test_method_c_training.py -v.
"""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'crack-test-matplotlib'))
import numpy as np
import torch
from src import data,method_c as c

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
P={**c.DEFAULT_PARAMS,'depth':1,'base_channels':2,'tile':32,'overlap':16,
   'batch_size':2,'max_epochs':3,'patience':10,'samples_per_epoch':4}
SPLIT={'train':['a','b'],'val':['v'],'test':['t']}
def pair(image_id):
    rng=np.random.default_rng({'a':1,'b':2,'v':3,'t':4}[image_id])
    image=rng.integers(80,200,(37,45,3),dtype=np.uint8)
    mask=np.zeros((37,45),bool);mask[5:32,20:24]=True;image[mask]=25
    return image,mask

def notebook_helpers():
    nb=json.loads((ROOT/'notebooks/train_unet_colab.ipynb').read_text(encoding='utf-8'))
    ns={}
    for name in ['cd-controls','cd-setup-helpers']:
        cell=next(cell for cell in nb['cells'] if cell.get('id')==name)
        exec(compile(''.join(cell['source']),name,'exec'),ns)
    return ns


class TrainingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.object(data,'load_split',return_value=SPLIT))
        self.stack.enter_context(patch.object(data,'load_pair',side_effect=pair))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    def tearDown(self):self.stack.close();self.tmp.cleanup()
    def train(self,path,**kw):return c.train_unet(SPLIT['train'],SPLIT['val'],P,path,'smoke',**kw)

    def test_continuous_equals_pause_resume_including_adam_rng(self):
        full=self.root/'full';part=self.root/'part'
        self.train(full)
        self.train(part,stop_after_epoch=1)
        before=c.read_training_checkpoint(part/'last.pt')
        self.assertEqual(before['epoch'],1)
        self.assertTrue(before['optimizer_state']['state'])
        result=self.train(part,resume_from=part/'last.pt')
        a=c.read_training_checkpoint(full/'last.pt');b=c.read_training_checkpoint(part/'last.pt')
        self.assertEqual(b['epoch'],3);self.assertEqual(result['status'],'completed')
        for key,value in a['model_state'].items():torch.testing.assert_close(value,b['model_state'][key],rtol=0,atol=0)
        for key,values in a['optimizer_state']['state'].items():
            for field,value in values.items():
                if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b['optimizer_state']['state'][key][field],rtol=0,atol=0)
        for field in ['torch_cpu','sampler_generator','loader_generator']:
            self.assertTrue(torch.equal(a['rng_state'][field],b['rng_state'][field]))
        self.assertEqual(a['rng_state']['python'],b['rng_state']['python'])
        self.assertEqual(a['rng_state']['numpy'],b['rng_state']['numpy'])
        self.assertEqual(a['best_epoch'],b['best_epoch'])
        self.assertEqual(a['epochs_without_gain'],b['epochs_without_gain'])
        for x,y in zip(a['history'],b['history']):
            for key in ['training_loss','validation_dice','validation_precision','validation_recall']:self.assertEqual(x[key],y[key])
        self.assertTrue((part/'history.csv').is_file());self.assertTrue((part/'figures/validation_dice.png').is_file())

    def test_best_and_early_stopping_survive_resume(self):
        path=self.root/'run'
        def score(dice):return {'f1':dice,'precision':.5,'recall':.5}
        p={**P,'patience':2}
        with patch.object(c,'_scores',return_value=score(.8)):
            c.train_unet(SPLIT['train'],SPLIT['val'],p,path,'smoke',stop_after_epoch=1)
        digest=c.file_sha256(path/'best.pt')
        with patch.object(c,'_scores',return_value=score(.7)):
            result=c.train_unet(SPLIT['train'],SPLIT['val'],p,path,'smoke',resume_from=path/'last.pt')
        self.assertEqual(c.file_sha256(path/'best.pt'),digest)
        self.assertEqual(result['best_epoch'],1);self.assertEqual(result['stopping_reason'],'patience')
        self.assertEqual(c.read_training_checkpoint(path/'last.pt')['epochs_without_gain'],2)

    def test_reject_incompatible_config_source_split_and_v1(self):
        path=self.root/'run';self.train(path,stop_after_epoch=1)
        digest=c.file_sha256(path/'best.pt')
        with self.assertRaisesRegex(ValueError,'configuration mismatch'):
            c.train_unet(SPLIT['train'],SPLIT['val'],{**P,'batch_size':1},path,'smoke',resume_from=path/'last.pt')
        with patch.object(c,'_source_fingerprint',return_value={'changed':'source'}),self.assertRaisesRegex(ValueError,'source code differs'):
            self.train(path,resume_from=path/'last.pt')
        original=c.file_sha256
        with patch.object(c,'file_sha256',side_effect=lambda p:'different' if Path(p)==c.config.SPLIT_FILE else original(p)),self.assertRaisesRegex(ValueError,'split hash differs'):
            self.train(path,resume_from=path/'last.pt')
        with self.assertRaises(FileExistsError):self.train(path)
        self.assertEqual(c.file_sha256(path/'best.pt'),digest)
        legacy=self.root/'legacy.pt';torch.save({'format_version':1},legacy)
        with self.assertRaisesRegex(ValueError,'v2'):c.read_training_checkpoint(legacy)

    def test_corrupt_alias_recovers_and_corrupt_snapshot_falls_back(self):
        path=self.root/'run';self.train(path,stop_after_epoch=1)
        (path/'last.pt').write_bytes(b'interrupted alias')
        self.assertEqual(c.read_training_checkpoint(path/'last.pt')['epoch'],1)
        (path/'best.pt').write_bytes(b'interrupted best alias')
        self.assertIsNotNone(c.load_unet(path/'best.pt','cpu'))
        receipt=c._verified_receipt(path)
        (path/receipt['checkpoint']).write_bytes(b'interrupted remote object')
        with self.assertWarns(UserWarning):
            state=c.read_training_checkpoint(path/'last.pt')
        self.assertEqual(state['epoch'],0)

    def test_formal_without_cuda_fails_before_creating_output(self):
        path=self.root/'forbidden'
        with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'requires CUDA'):
            c.train_unet(SPLIT['train'],SPLIT['val'],P,path,'formal')
        self.assertFalse(path.exists())

    def test_detected_but_broken_cuda_fails(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.arange',side_effect=RuntimeError('driver broken')):
            with self.assertRaisesRegex(RuntimeError,'GPU tensor operation failed'):c.check_run_mode('formal')

    def test_failed_publication_preserves_previous_file(self):
        destination=self.root/'log.json';destination.write_text('old',encoding='utf-8')
        with patch.object(c.os,'replace',side_effect=OSError('upload interrupted')):
            with self.assertRaises(OSError):c._write_json(destination,{'new':'log'})
        self.assertEqual(destination.read_text(),'old')


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.ns=notebook_helpers()
        self.split=self.root/'split.json';self.split.write_text(json.dumps({'train':['a'],'val':['b'],'test':[]}),encoding='utf-8')
        self.archive=self.root/'dataset.zip'
        with zipfile.ZipFile(self.archive,'w') as z:
            for folder in ['images','masks']:
                for image_id in ['a','b']:z.writestr(folder+'/'+image_id+'.jpg',b'unchanged fixture bytes')
        self.counts={'train':1,'val':1,'test':0}
    def tearDown(self):self.tmp.cleanup()
    def prepare(self,dest):return self.ns['prepare_dataset'](self.archive,dest,self.split,self.counts)

    def test_idempotent_data_preparation_and_modified_data_rejection(self):
        dest=self.root/'data';first=self.prepare(dest)
        original=(dest/'images/a.jpg').read_bytes();stamp=(dest/'images/a.jpg').stat().st_mtime_ns
        second=self.prepare(dest);self.assertEqual(first,second)
        self.assertEqual((dest/'images/a.jpg').read_bytes(),original)
        self.assertEqual((dest/'images/a.jpg').stat().st_mtime_ns,stamp)
        (dest/'images/a.jpg').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'modified'):self.prepare(dest)
        self.assertEqual((dest/'images/a.jpg').read_bytes(),b'modified')

    def test_unknown_existing_directory_not_overwritten(self):
        dest=self.root/'data';dest.mkdir();(dest/'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):self.prepare(dest)
        self.assertEqual((dest/'keep.txt').read_text(),'keep')

    def test_zip_traversal_rejected(self):
        with zipfile.ZipFile(self.archive,'a') as z:z.writestr('../escape.jpg',b'bad')
        with self.assertRaisesRegex(ValueError,'Unsafe ZIP'):self.prepare(self.root/'data')
        self.assertFalse((self.root/'escape.jpg').exists())

    def test_latest_uploaded_code_gets_distinct_verified_snapshot(self):
        source=self.root/'uploaded';source.mkdir()
        required=['src/method_c.py','src/method_d.py','src/config.py','src/data.py','src/utils.py',
          'src/evaluate.py','src/measure.py','src/__init__.py','splits/splits.json','requirements.txt']
        for name in required:
            p=source/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text('uploaded v1',encoding='utf-8')
        first,_=self.ns['stage_uploaded_code'](source,self.root/'code')
        (source/'src/method_c.py').write_text('uploaded v2',encoding='utf-8')
        second,manifest=self.ns['stage_uploaded_code'](source,self.root/'code')
        self.assertNotEqual(first,second);self.assertEqual((first/'src/method_c.py').read_text(),'uploaded v1')
        self.assertEqual((second/'src/method_c.py').read_text(),'uploaded v2')

    def test_all_notebook_cells_compile_and_formal_default_is_explicit(self):
        nb=json.loads((ROOT/'notebooks/train_unet_colab.ipynb').read_text(encoding='utf-8'))
        ids=[cell['id'] for cell in nb['cells']];self.assertEqual(len(ids),len(set(ids)))
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')
        self.assertEqual(self.ns['RUN_MODE'],'formal')
        text=''.join(''.join(cell['source']) for cell in nb['cells'])
        self.assertNotIn('SMOKE = not torch.cuda.is_available()',text)
        self.assertNotIn('--index-url https://download.pytorch.org/whl/cpu',text)


if __name__=='__main__':unittest.main(verbosity=2)
