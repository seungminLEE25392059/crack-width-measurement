"""Lingyu Wu: execute unchanged team validation entry points with verified C probability replay.

Inference is performed once on ALL 69 originals by run_method_cd_validation.py.
This adapter replays those EXACT float32 maps to the team's C export entry point;
it changes no shared source or predictions. Team C timing measures cache export,
so original inference timings are retained separately and explicitly labelled.
"""
from __future__ import annotations
import argparse,contextlib,csv,json,os,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from src import config,data,method_c as c,method_d as d,utils
from run_method_cd_validation import read,save,digest,checked_cache,array_sha,csv_write,MODEL_SHA

class ProbabilityReplay:
    def __init__(self,work,protocol,manifest):
        self.work,self.protocol,self.manifest=work,protocol,manifest
        self.lookup={entry['decoded_image_sha256']:image_id for image_id,entry in manifest['images'].items()}
        if len(self.lookup)!=len(manifest['images']):raise ValueError('Ambiguous identical image hashes in replay')
        self.seen=[]
    def __call__(self,image,net,params):
        if net.checkpoint_metadata['epoch']!=19:raise ValueError('Replay model epoch differs')
        for key,value in self.protocol['inference_params'].items():
            if params[key]!=value:raise ValueError('Replay inference setting differs: '+key)
        key=array_sha(image)
        if key not in self.lookup:raise ValueError('Image not in exact decoded validation cache')
        image_id=self.lookup[key];prob,entry=checked_cache(self.work,image_id,self.manifest)
        if list(image.shape[:2])!=entry['shape']:raise ValueError('Replay dimensions differ')
        self.seen.append(image_id)
        return prob>=params['threshold'],prob

class Tee:
    def __init__(self,terminal,file):self.terminal,self.file=terminal,file
    def write(self,value):self.terminal.write(value);self.file.write(value);self.file.flush()
    def flush(self):self.terminal.flush();self.file.flush()

def entry(script,args,log):
    saved=sys.argv[:]
    try:
        sys.argv=[str(ROOT/'scripts'/script),*args]
        with Path(log).open('x',encoding='utf-8') as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)):
            try:
                runpy.run_path(str(ROOT/'scripts'/script),run_name='__main__')
            except SystemExit as stopped:
                # Public CLI entry points may use sys.exit(main()). A successful
                # CLI exit must not silently skip the following validation stage.
                if stopped.code not in (None,0):
                    raise
    finally:sys.argv=saved

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--verify-existing-exports',action='store_true',
        help='Keep existing masks; verify all against frozen probabilities/params before finishing evaluation')
    args=parser.parse_args();work=args.work_dir.resolve();os.environ['MPLCONFIGDIR']=str(work/'matplotlib_cache')
    os.environ['MPLBACKEND']='Agg'
    import torch,cv2
    torch.set_num_threads(read(work/'model_export.json')['selected_threads']);cv2.setNumThreads(1)
    protocol=read(work/'validation_protocol.json');manifest=read(work/'probability_cache_manifest.json')
    assert digest(ROOT/'models/unet.pt')==MODEL_SHA and len(manifest['images'])==69
    assert c._source_fingerprint()==protocol['source_fingerprints']
    ids=data.load_split()['val'];assert ids==protocol['validation_ids']
    assert read(config.PARAMS_DIR/'C.json')==read(work/'threshold_search.json')['selected']
    d_params=read(config.PARAMS_DIR/'D.json');assert d_params==read(work/'d_search.json')['params']
    for method in ('C','D'):
        target=config.RESULTS_DIR/method/'val'
        exists=target.exists() and any(target.iterdir())
        if exists and not args.verify_existing_exports:raise FileExistsError('Existing team results preserved: '+str(target))
        if args.verify_existing_exports:
            if not exists:raise FileNotFoundError('Cannot resume absent exports: '+str(target))
            expected={**(c.DEFAULT_PARAMS if method=='C' else d.DEFAULT_PARAMS),**read(config.PARAMS_DIR/f'{method}.json')}
            assert read(target/'params_used.json')==expected,'Exported parameters differ'
    eval_dir=config.EVAL_DIR/'val'
    if eval_dir.exists() and any(eval_dir.rglob('*')):raise FileExistsError('Existing shared evaluation outputs preserved')
    env_path=config.LOG_DIR/'environment.txt'
    original_log_dir=config.LOG_DIR
    if env_path.exists():
        # Preserve preexisting group log; only redirect this generated artifact.
        config.LOG_DIR=work/'team_log'
    source_hashes={str(path.relative_to(ROOT)):digest(path) for path in (ROOT/'scripts').glob('*.py')}
    replay=ProbabilityReplay(work,protocol,manifest);original=c.segment_c
    if not args.verify_existing_exports:
        try:
            c.segment_c=replay
            entry('run_method.py',['--method','C','--split','val'],work/'team_run_C.log')
        finally:c.segment_c=original
        assert replay.seen==ids
        inference_rows=[{'image_id':image_id,'seconds':manifest['images'][image_id]['inference_seconds']} for image_id in ids]
        csv_write(config.RESULTS_DIR/'C/val/inference_timing_original.csv',inference_rows)
        save(config.RESULTS_DIR/'C/val/timing_provenance.json',{
            'timing.csv':'actual team export call timing with checksum-verified float32 probability replay; NOT fresh U-Net compute',
            'inference_timing_original.csv':'actual first complete native-image U-Net inference timings on CPU',
            'probabilities_quantized_for_selection':False,'all_69_originals_inferred':True,'model_sha256':MODEL_SHA})
        entry('run_method.py',['--method','D','--split','val'],work/'team_run_D.log')
    else:
        assert read(config.RESULTS_DIR/'C/val/timing_provenance.json')['model_sha256']==MODEL_SHA
        verified_net=c.load_unet(ROOT/'models/unet.pt',device='cpu')
    checked=[]
    for method in ('C','D'):
        folder=config.RESULTS_DIR/method/'val'
        actual={path.stem for path in folder.glob('*.png') if not path.stem.endswith('_prob')}
        assert actual==set(ids) and len(actual)==69
        for image_id in ids:
            raw=utils._imread(folder/f'{image_id}.png',cv2.IMREAD_GRAYSCALE,'exported mask')
            assert set(np.unique(raw).tolist())<={0,255}
            ref=utils.read_mask(data._index(str(config.MASK_DIR))[image_id]);assert raw.shape==ref.shape
            if method=='C' and args.verify_existing_exports:
                image,_=data.load_pair(image_id)
                expected,_=replay(image,verified_net,read(config.PARAMS_DIR/'C.json'))
                assert np.array_equal(raw>0,expected),'Existing C export differs from exact frozen probabilities'
            if method=='D':
                mask_c=utils.read_mask(utils.result_path('C','val',image_id))
                assert np.array_equal(raw>0,d.refine_d(mask_c,d_params))
            checked.append({'method':method,'image_id':image_id,'height':raw.shape[0],'width':raw.shape[1],
                            'sha256':digest(folder/f'{image_id}.png'),'foreground_pixels':int((raw>0).sum())})
    csv_write(work/'tables/exported_masks_manifest.csv',checked)
    if args.verify_existing_exports:assert replay.seen==ids
    interface_log=work/('team_interfaces_rechecked.log' if args.verify_existing_exports else 'team_interfaces_final.log')
    entry('check_interfaces.py',[],interface_log)
    try:entry('run_evaluation.py',['--split','val','--methods','C,D'],work/'team_evaluation.log')
    finally:config.LOG_DIR=original_log_dir
    assert all(digest(ROOT/name)==sha for name,sha in source_hashes.items())
    report={'method_C_masks':69,'method_D_masks':69,'all_dimensions_and_binary_encoding_verified':True,
        'D_exactly_refines_exported_C':True,'team_entry_points_executed':['run_method.py C val','run_method.py D val',
            'check_interfaces.py','run_evaluation.py --split val --methods C,D'],
        'shared_scripts_unchanged':True,'C_export_cache_replay_explained':True,
        'final_D_selection_feasible':d_params['selection_feasible'],'test_or_final_executed':False,
        'resumed_after_existing_exports':args.verify_existing_exports}
    save(work/'team_validation.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
