"""Execute every saved Notebook code cell for a local real-data CPU smoke run.

Usage: .venv/Scripts/python -B tests/run_method_cd_smoke.py
Requires existing dataset/venv. Creates a fresh eval/method_cd run; no final test.
Owner: Lingyu Wu. No Jupyter runtime is needed; cells contain ordinary Python.
"""
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['CRACK_PROJECT_DIR']=str(ROOT)
os.environ['MPLBACKEND']='Agg'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
nb=json.loads((ROOT/'notebooks/train_unet_colab.ipynb').read_text(encoding='utf-8-sig'))
namespace={'__name__':'__notebook__'}
executed=[]
transcript=io.StringIO()
class Tee:
    def write(self,s):
        sys.__stdout__.write(s);sys.__stdout__.flush();transcript.write(s)
    def flush(self):sys.__stdout__.flush()
try:
    with contextlib.redirect_stdout(Tee()):
        for index,cell in enumerate(nb['cells']):
            if cell['cell_type']!='code':continue
            print(f'Executing notebook cell {index}',flush=True)
            start=time.perf_counter()
            exec(compile(''.join(cell['source']),f'notebook:cell-{index}','exec'),namespace)
            if cell.get('id') == 'cd-controls':
                # This test explicitly selects smoke; production defaults formal.
                namespace.update(RUN_MODE='smoke',RUN_ACTION='new',RUN_BENCHMARK=True,RUN_POSTPROCESS=True)
            executed.append({'cell_index':index,'status':'passed','seconds':time.perf_counter()-start})
        # Exercise the actual Notebook resume controls on a completed run. It
        # must report completion without rerunning an epoch or changing best.pt.
        from src import method_c
        digest = method_c.file_sha256(namespace['OUT']/'training/best.pt')
        namespace.update(RUN_ACTION='resume',RESUME_RUN_ID=namespace['RUN_ID'])
        for name in ['cd-config','cd-progress','cd-train']:
            cell=next(c for c in nb['cells'] if c.get('id')==name)
            exec(compile(''.join(cell['source']),name,'exec'),namespace)
        assert method_c.file_sha256(namespace['OUT']/'training/best.pt') == digest
        # Exercise existing CLI entry points in an isolated output configuration.
        # No team model/params/results path is changed on disk.
        from src import config, utils
        integration = namespace['OUT']/'integration'
        original = config.MODELS_DIR, config.PARAMS_DIR, config.RESULTS_DIR
        original_argv = sys.argv[:]
        try:
            config.MODELS_DIR=integration/'models'
            config.PARAMS_DIR=integration/'params'
            config.RESULTS_DIR=integration/'results'
            config.MODELS_DIR.mkdir(parents=True)
            shutil.copy2(namespace['OUT']/'training/best.pt', config.MODELS_DIR/'unet.pt')
            utils.save_params('C', namespace['c_params'])
            utils.save_params('D', namespace['d_params'])
            checker=runpy.run_path(str(ROOT/'scripts/check_interfaces.py'),run_name='interface_test')
            assert checker['check_method_c']()[0]
            for method in ['C','D']:
                sys.argv=['run_method.py','--method',method,'--split','val','--limit','1']
                runpy.run_path(str(ROOT/'scripts/run_method.py'),run_name='__main__')
            (integration/'status.json').write_text(json.dumps({'C_existing_runner':'passed','D_existing_runner':'passed',
            'C_interface_with_real_checkpoint':'passed','scope':'one validation image; smoke weights'},indent=2),encoding='utf-8')
        finally:
            config.MODELS_DIR,config.PARAMS_DIR,config.RESULTS_DIR=original
            sys.argv=original_argv
    output=namespace['OUT']
    (output/'notebook_execution.json').write_text(json.dumps({'cells':executed,'status':'passed','execution':'ordinary Python cells; not a Jupyter UI run'},indent=2),encoding='utf-8')
    (output/'notebook_execution.txt').write_text(transcript.getvalue(),encoding='utf-8')
    from src import method_c
    manifest=[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':method_c.file_sha256(p)}
              for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json']
    method_c._write_json(output/'manifest.json',{'artifacts':manifest})
    print('NOTEBOOK_OUTPUT='+str(output),flush=True)
finally:
    if 'OUT' in namespace:
        (namespace['OUT']/'notebook_execution.txt').write_text(transcript.getvalue(),encoding='utf-8')
