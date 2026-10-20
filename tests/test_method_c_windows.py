"""Transfer tests: warm start resets progress; strict resume remains strict."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src import data,method_c as c
from test_method_c_training import P,SPLIT,pair

class WindowsTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.object(data,'load_split',return_value=SPLIT))
        self.stack.enter_context(patch.object(data,'load_pair',side_effect=pair))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    def tearDown(self):self.stack.close();self.tmp.cleanup()
    def train(self,path,**kw):return c.train_unet(SPLIT['train'],SPLIT['val'],P,path,'smoke',**kw)
    def source(self):
        source=self.root/'source';self.train(source,stop_after_epoch=1)
        return source/'best.pt'
    def test_warm_start_exact_initial_weights_but_fresh_adam_history(self):
        path=self.source();sha=c.file_sha256(path);destination=self.root/'warm'
        original=torch.load(path,map_location='cpu',weights_only=True)
        self.train(destination,stop_after_epoch=1,warm_start_from=path,warm_start_sha256=sha)
        receipt=json.loads(next((destination/'commits').glob('epoch_000000_*.json')).read_text())
        epoch0=torch.load(destination/receipt['checkpoint'],map_location='cpu',weights_only=True)
        for key,value in original['model_state'].items():torch.testing.assert_close(value,epoch0['model_state'][key],rtol=0,atol=0)
        self.assertEqual(epoch0['epoch'],0);self.assertEqual(epoch0['history'],[])
        self.assertEqual(epoch0['optimizer_state']['state'],{})
        self.assertEqual(epoch0['epochs_without_gain'],0);self.assertEqual(epoch0['best_val_dice'],-1)
        self.assertEqual(epoch0['log']['initialization']['source_sha256'],sha)
        self.assertFalse(epoch0['log']['initialization']['strict_resume'])
        resumed=self.train(destination,resume_from=destination/'last.pt')
        self.assertEqual(resumed['status'],'completed')
        self.assertEqual(resumed['initialization']['source_sha256'],sha)
    def test_bad_hash_architecture_split_and_smoke_source_rejected(self):
        source=self.source();sha=c.file_sha256(source)
        with self.assertRaisesRegex(ValueError,'SHA256'):self.train(self.root/'bad',warm_start_from=source,warm_start_sha256='0'*64)
        self.assertFalse((self.root/'bad').exists())
        with self.assertRaisesRegex(ValueError,'architecture'):c.read_warm_start(source,{**P,'base_channels':4},SPLIT['train'],SPLIT['val'],sha)
        with patch.object(c.config,'SPLIT_FILE',source),self.assertRaisesRegex(ValueError,'split hash'):c.read_warm_start(source,P,SPLIT['train'],SPLIT['val'],sha)
        # Formal core rejects smoke origin before any output/epoch is created.
        formal={**c.DEFAULT_PARAMS,**c.FORMAL_INITIAL_PARAMS}
        state=torch.load(source,map_location='cpu',weights_only=True);state['params']=formal
        candidate=self.root/'smoke-labelled.pt';torch.save(state,candidate)
        full={'train':[str(i) for i in range(320)],'val':[str(i) for i in range(320,389)],'test':[]}
        state['log']['train_ids']=full['train'];state['log']['val_ids']=full['val'];torch.save(state,candidate)
        with patch.object(data,'load_split',return_value=full),patch.object(c,'check_run_mode',return_value={'device':'cuda'}):
            with self.assertRaisesRegex(ValueError,'smoke weights'):
                c.train_unet(full['train'],full['val'],formal,self.root/'not-formal','formal',warm_start_from=candidate,warm_start_sha256=c.file_sha256(candidate))
        self.assertFalse((self.root/'not-formal').exists())
    def test_mutually_exclusive_and_missing_git_are_safe(self):
        source=self.source()
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            self.train(self.root/'both',resume_from=source,warm_start_from=source)
        with patch.object(c.subprocess,'run',side_effect=FileNotFoundError('git not installed')):
            self.train(self.root/'no git',stop_after_epoch=1)
        self.assertEqual(c.read_training_checkpoint(self.root/'no git/last.pt')['log']['git_revision'],'')

if __name__=='__main__':unittest.main(verbosity=2)
