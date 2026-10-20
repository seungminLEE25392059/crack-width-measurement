"""Lingyu Wu: resumable FULL-validation width search through existing Method D code.

Calls Junsen's public measurement and the public matcher via method_d. No private
width algorithm. No Test split. Failed recall/miss/bias constraints stay visible.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from src import config,data,method_c as c,method_d as d,measure,utils
from run_method_cd_validation import read,save,csv_write,digest,checked_cache,MODEL_SHA

AREAS=(0,10,50);ELONGATIONS=(1.,2.,3.)
CONSTRAINTS={'max_recall_drop':.02,'max_extra_misses':0,'max_negative_bias':1.}
SUM_KEYS=('detection_failures','reference_cracks','matched_width_samples','false_detections','fragments',
          'merged_predictions','merged_extra_references','zero_length_references')
WIDTH_KEYS=('mean_width_mae','max_width_mae','signed_width_error','max_width_bias',
            'relative_length_error','signed_relative_length_error')

def aggregate_image_rows(images,names):
    rows=[]
    for name in names:
        group=[row for row in images if row['variant']==name]
        if not group:raise ValueError('Missing variant '+name)
        counts=np.array([sum(int(row[k]) for row in group) for k in ('tp','fp','fn')],np.int64)
        n=sum(row['matched_width_samples'] for row in group)
        row={'variant':name,**c._scores(counts),'tp':int(counts[0]),'fp':int(counts[1]),'fn':int(counts[2])}
        for k in WIDTH_KEYS:
            row[k]=sum(r[k]*r['matched_width_samples'] for r in group if r[k] is not None)/n if n else None
        row.update({k:sum(int(r[k]) for r in group) for k in SUM_KEYS})
        row.update(postprocessing_seconds=sum(r['postprocessing_seconds'] for r in group)/len(group),
                   measurement_status='shared',validation_images=len(group))
        rows.append(row)
    return rows

def evaluate_resumable(work,ids,threshold,variants,label):
    folder=work/label;folder.mkdir(exist_ok=True)
    protocol={'validation_ids':ids,'threshold':threshold,'variants':variants,
        'model_sha256':MODEL_SHA,'measure_sha256':digest(ROOT/'src/measure.py'),
        'evaluate_sha256':digest(ROOT/'src/evaluate.py'),'method_d_sha256':digest(ROOT/'src/method_d.py'),
        'probability_manifest_sha256':digest(work/'probability_cache_manifest.json'),
        'matching':'unchanged public overlap matcher; report misses/false detections/fragments/many-to-one merges',
        'measurement':'unchanged Junsen widths and skeleton lengths, pixels; matched-only errors accompany coverage counts'}
    path=folder/'protocol.json'
    if path.exists():assert read(path)==protocol,'Width protocol changed; old results preserved'
    else:save(path,protocol)
    manifest=read(work/'probability_cache_manifest.json');all_rows=[];per_image=[]
    for index,image_id in enumerate(ids,1):
        assert digest(ROOT/'src/measure.py')==protocol['measure_sha256']
        assert digest(ROOT/'src/method_d.py')==protocol['method_d_sha256']
        out=folder/f'{image_id}.json'
        if out.exists():record=read(out);assert record['protocol_sha256']==digest(path)
        else:
            checked_cache(work,image_id,manifest)
            tick=time.perf_counter()
            rows,details=d.evaluate_variants([image_id],work/'probabilities',threshold,d.DEFAULT_PARAMS,variants)
            record={'image_id':image_id,'protocol_sha256':digest(path),'rows':rows,'per_image':details,
                    'seconds':time.perf_counter()-tick}
            save(out,record)
        all_rows.extend(record['rows']);per_image.extend(record['per_image'])
        print(f'{label.upper()} [{index}/69] {image_id}: {record["seconds"]:.2f}s',flush=True)
        save(work/'width_progress.json',{'stage':label,'images_completed':index,'images_total':len(ids),'latest_id':image_id})
    rows=aggregate_image_rows(all_rows,list(variants))
    assert all(row['validation_images']==69 for row in rows)
    csv_write(work/'tables'/f'{label}.csv',rows);csv_write(work/'tables'/f'{label}_per_image.csv',per_image)
    return rows,per_image,protocol

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args();work=args.work_dir.resolve();os.environ['MPLCONFIGDIR']=str(work/'matplotlib_cache')
    import cv2
    cv2.setNumThreads(1)
    selected=read(work/'threshold_search.json')['selected'];threshold=selected['threshold'];ids=data.load_split()['val']
    assert len(ids)==69 and selected['selection_ids']==ids and selected['selection_complete']
    assert not utils.is_placeholder(measure.measure_cracks) and hasattr(measure,'CRACK_KEYS')
    variants={'Baseline':{**d.DEFAULT_PARAMS,'use_filter':False,'use_erosion':False}}
    variants.update({f'area={a},ratio={r}':{**d.DEFAULT_PARAMS,'min_area':a,'min_elongation':r}
                     for a in AREAS for r in ELONGATIONS})
    rows,per_image,protocol=evaluate_resumable(work,ids,threshold,variants,'D_width_search')
    winner=d.select_width_candidate(rows,**CONSTRAINTS)
    baseline=rows[0]
    for row in rows:
        row['recall_drop_vs_C']=baseline['recall']-row['recall']
        row['extra_misses_vs_C']=row['detection_failures']-baseline['detection_failures']
        row['constraint_eligible']=(row['variant']!='Baseline' and row['recall']>=baseline['recall']-.02
            and row['detection_failures']<=baseline['detection_failures']
            and row['max_width_mae'] is not None and row['max_width_bias']>=-1.)
    # Preserve an explicit diagnostic configuration if ALL candidates fail.
    diagnostic=min((r for r in rows[1:] if r['max_width_mae'] is not None),
                   key=lambda r:(r['max_width_mae'],r['mean_width_mae'],-r['recall']),default=None)
    chosen=winner or diagnostic
    if chosen is None:raise RuntimeError('No matched shared-width samples: D selection blocked, no D params published')
    p=variants[chosen['variant']]
    params={**p,'threshold_c':threshold,'model_path':'models/unet.pt','model_sha256':MODEL_SHA,
        'selection_source':'fixed validation split only','selection_ids':ids,
        'selection_feasible':winner is not None,'selection_complete':winner is not None,
        'validation_complete':True,'parameter_search_complete':True,
        'selection_status':'width_selected' if winner else 'diagnostic_only_no_candidate_satisfies_original_constraints',
        'deployment_recommended':bool(winner and chosen['max_width_mae']<baseline['max_width_mae']
                                      and chosen['mean_width_mae']<=baseline['mean_width_mae']),
        'comparison_to_C':{'max_width_mae_change':chosen['max_width_mae']-baseline['max_width_mae'],
            'mean_width_mae_change':chosen['mean_width_mae']-baseline['mean_width_mae'],
            'signed_mean_width_error_change':chosen['signed_width_error']-baseline['signed_width_error'],
            'signed_max_width_error_change':chosen['max_width_bias']-baseline['max_width_bias'],
            'recall_change':chosen['recall']-baseline['recall'],
            'missed_crack_change':chosen['detection_failures']-baseline['detection_failures'],
            'fragment_change':chosen['fragments']-baseline['fragments'],
            'relative_length_mae_change':chosen['relative_length_error']-baseline['relative_length_error']},
        'constraints':CONSTRAINTS,
        'selection_metric':'max width MAE, mean width MAE, recall tie-break; public matched-only errors with coverage guards',
        'selected_validation_scores':chosen,'measure_sha256':protocol['measure_sha256'],
        'erosion_footprint':d.EROSION_FOOTPRINT.astype(int).tolist(),'erosion_border_value':1}
    result={'protocol':protocol,'constraints':CONSTRAINTS,'params':params,'candidates':rows,
        'scope':'all 69 validation images; public shared widths; no Test use',
        'diagnostic_warning':None if winner else 'NO SAFE D CANDIDATE under existing constraints. Diagnostic masks/ablations may be exported, but not claimed as validated deployment parameters.'}
    save(work/'d_search.json',result);csv_write(work/'tables/D_width_search.csv',rows)
    ablations={name:{**p,'use_filter':f,'use_erosion':e} for name,(f,e) in d.ABLATIONS.items()}
    ab_rows,ab_details,_=evaluate_resumable(work,ids,threshold,ablations,'D_ablation')
    save(work/'d_ablation.json',{'params':params,'rows':ab_rows,'per_image':ab_details})
    # D.json is explicitly labelled diagnostic when the original guards fail;
    # the shared runner can still export honest experimental masks for reporting.
    path=config.PARAMS_DIR/'D.json'
    if path.exists() and read(path)!=params:raise FileExistsError('Existing different D params preserved')
    utils.save_params('D',params)
    save(work/'width_progress.json',{'stage':'complete','validation_images':69,'selection_feasible':winner is not None})
    print('D WIDTH SELECTION',json.dumps({'feasible':winner is not None,'selected':chosen,'params':params}),flush=True)

if __name__=='__main__':main()
