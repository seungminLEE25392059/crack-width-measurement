"""Lingyu Wu: full fixed-validation inference/selection without changing shared code.

Input: frozen best model, fixed split, original images/masks. No Test execution.
Output: lossless probability cache, auditable threshold search and C parameters.
This is validation only; no optimizer or training is started.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
import numpy as np
from src import config,data,evaluate,method_c as c,utils

MODEL_SHA='bfbc271099db4a6d928d7766a9f114048e802b01973ead689d440979d581ff08'
DATA_SHA='65e863ecce9d6302618945760e9765cb162ac76b297bf1638f96e2d3693a846a'
THRESHOLDS=(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99)
BASELINE={'f1':.7939032112397485,'precision':.6847368791740469,'recall':.944479890365602}
BASELINE_TOLERANCE=5e-6

def digest(path):return c.file_sha256(Path(path))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):c._write_json(Path(path),value)
def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def checked_cache(work,image_id,expected):
    entry=expected['images'][image_id];path=work/'probabilities'/f'{image_id}_prob.npy'
    if digest(path)!=entry['sha256']:raise ValueError('Lossless probability cache SHA mismatch: '+image_id)
    prob=np.load(path,allow_pickle=False)
    if prob.dtype!=np.float32 or list(prob.shape)!=entry['shape'] or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:
        raise ValueError('Invalid probability map: '+image_id)
    return prob,entry
def select_threshold(rows):
    return max(rows,key=lambda row:(row['f1'],row['recall'],-abs(row['threshold']-.5)))

def infer(work):
    import torch
    work.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(work/'matplotlib_cache')
    export=read(work/'model_export.json');torch.set_num_threads(export['selected_threads'])
    model=ROOT/'models/unet.pt'
    assert digest(model)==MODEL_SHA
    net=c.load_unet(model,device='cpu');p=net.unet_params
    split=data.load_split();ids=split['val']
    assert len(ids)==69 and len(split['train'])==320 and len(split['test'])==69
    assert not set(ids)&set(split['test']) and not set(ids)&set(split['train'])
    reference=read(ROOT.parent/'method_cd_rtx5070ti_package/manifests/dataset_manifest.json')
    names={}
    for image_id in ids:
        for folder,root in [('images',config.IMAGE_DIR),('masks',config.MASK_DIR)]:
            path=data._index(str(root))[image_id];name=folder+'/'+path.name
            assert digest(path)==reference['files'][name]['sha256'],name
            names[name]=reference['files'][name]['sha256']
    # The package's immutable original-byte manifest supplies all 458 pairs.
    all_hashes={name:info['sha256'] for name,info in reference['files'].items()}
    assert hashlib.sha256(json.dumps(all_hashes,sort_keys=True).encode()).hexdigest()==DATA_SHA
    protocol={'schema':1,'model_sha256':MODEL_SHA,'best_epoch':19,'source_fingerprints':c._source_fingerprint(),
        'split_sha256':digest(config.SPLIT_FILE),'validation_ids':ids,'dataset_fingerprint':DATA_SHA,
        'inference_params':{key:p[key] for key in ('depth','base_channels','tile','overlap','batch_size','normalization')},
        'environment':c.environment_info(),'inference_device':'cpu','torch_threads':torch.get_num_threads(),
        'threshold_candidates':list(THRESHOLDS),'selection_metric':'micro full-image foreground Dice; recall tie-break; closeness to 0.5 tie-break',
        'baseline_reference':BASELINE,'baseline_absolute_tolerance':BASELINE_TOLERANCE,
        'no_training_or_test_evaluation':True,'float_probability_format':'float32 npy; no quantization'}
    protocol_path=work/'validation_protocol.json'
    if protocol_path.exists():assert read(protocol_path)==protocol,'Validation protocol changed; preserve old output and review'
    else:save(protocol_path,protocol)
    cache_dir=work/'probabilities';cache_dir.mkdir(exist_ok=True)
    manifest_path=work/'probability_cache_manifest.json'
    manifest=read(manifest_path) if manifest_path.exists() else {'protocol_sha256':digest(protocol_path),'images':{}}
    assert manifest['protocol_sha256']==digest(protocol_path)
    totals={threshold:np.zeros(3,np.int64) for threshold in THRESHOLDS};pixels=0;gt_pixels=0;per_image=[]
    for index,image_id in enumerate(ids,1):
        assert c._source_fingerprint()==protocol['source_fingerprints'],'Source changed during validation'
        started=time.perf_counter();image,ref=data.load_pair(image_id)
        if image_id in manifest['images']:
            prob,entry=checked_cache(work,image_id,manifest)
            assert entry['decoded_image_sha256']==array_sha(image) and list(ref.shape)==entry['shape']
        else:
            target=cache_dir/f'{image_id}_prob.npy'
            if target.exists():raise FileExistsError('Unregistered probability file preserved: '+str(target))
            tick=time.perf_counter();_,prob=c.segment_c(image,net,p);seconds=time.perf_counter()-tick
            assert prob.shape==ref.shape and prob.dtype==np.float32 and np.isfinite(prob).all()
            assert prob.min()>=0 and prob.max()<=1
            pending=target.with_suffix('.npy.pending')
            with pending.open('xb') as stream:
                np.save(stream,prob,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            os.replace(pending,target)
            entry={'sha256':digest(target),'shape':list(prob.shape),'dtype':'float32','inference_seconds':seconds,
                   'decoded_image_sha256':array_sha(image),'probability_range':[float(prob.min()),float(prob.max())]}
            manifest['images'][image_id]=entry;save(manifest_path,manifest)
        pixels+=ref.size;gt_pixels+=int(ref.sum())
        for threshold in THRESHOLDS:
            counts=c._strict_counts(prob>=threshold,ref);totals[threshold]+=counts
            per_image.append({'image_id':image_id,'threshold':threshold,**c._scores(counts),
                'tp':int(counts[0]),'fp':int(counts[1]),'fn':int(counts[2]),'pixels':ref.size,
                'predicted_foreground_fraction':float((counts[0]+counts[1])/ref.size),
                'inference_seconds':entry['inference_seconds']})
        save(work/'progress.json',{'stage':'C_full_validation','images_completed':index,'images_total':69,
             'latest_id':image_id,'cached_float32_maps':len(manifest['images'])})
        print(f'FULL VALIDATION [{index}/69] {image_id}: inference {entry["inference_seconds"]:.2f}s; wall {time.perf_counter()-started:.2f}s',flush=True)
        del image,ref,prob
    rows=[{'threshold':threshold,**c._scores(counts),'tp':int(counts[0]),'fp':int(counts[1]),'fn':int(counts[2]),
           'predicted_foreground_fraction':float((counts[0]+counts[1])/pixels),'validation_images':69}
          for threshold,counts in totals.items()]
    reproduced=next(row for row in rows if row['threshold']==.5)
    differences={key:reproduced[key]-value for key,value in BASELINE.items()}
    baseline_ok=all(abs(value)<=BASELINE_TOLERANCE for value in differences.values())
    result={'protocol':protocol,'candidates':rows,'validation_pixels':pixels,'gt_foreground_pixels':gt_pixels,
            'baseline_0_5':{'reference':BASELINE,'actual':reproduced,'differences':differences,
                           'absolute_tolerance':BASELINE_TOLERANCE,'pass':baseline_ok},
            'selection_status':'selected' if baseline_ok else 'BLOCKED_baseline_reproduction_review_required'}
    csv_write(work/'tables/C_threshold_search.csv',rows);csv_write(work/'tables/C_threshold_per_image.csv',per_image)
    save(work/'threshold_search.json',result)
    if not baseline_ok:raise RuntimeError('0.5 baseline differs beyond declared tolerance. Cached maps retained; no final C params written.')
    selected=select_threshold(rows)
    chosen={**p,'threshold':selected['threshold'],'model_path':'models/unet.pt','model_sha256':MODEL_SHA,
            'best_epoch':19,'training_run_id':'rtx5070ti_formal_001','selection_source':'fixed validation split only',
            'selection_metric':protocol['selection_metric'],'selection_ids':ids,'selection_complete':True,
            'threshold_candidates':list(THRESHOLDS),'selected_validation_scores':selected,
            'split_sha256':protocol['split_sha256'],'training_source_fingerprints':protocol['source_fingerprints']}
    params_path=config.PARAMS_DIR/'C.json'
    if params_path.exists() and read(params_path)!=chosen:raise FileExistsError('Existing C parameters differ and were preserved')
    saved=utils.save_params('C',chosen);assert read(saved)==chosen
    result['selected']=chosen;save(work/'threshold_search.json',result)
    save(work/'progress.json',{'stage':'C_threshold_selection_complete','images_completed':69,'selected_threshold':chosen['threshold']})
    print('C SELECTED',json.dumps(selected),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['infer'])
    parser.add_argument('--work-dir',type=Path,required=True);args=parser.parse_args()
    infer(args.work_dir.resolve())
if __name__=='__main__':main()
