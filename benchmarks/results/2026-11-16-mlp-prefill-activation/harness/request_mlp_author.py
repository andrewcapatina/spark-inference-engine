"""Local-model implementation only; inspect legacy runtime before authoring."""
from pathlib import Path
import sys,subprocess
from nvfp4_update_runtime import Runtime,R,P,C,M
run=Path((R/'feedback/mlp-main-run.txt').read_text().strip());number=sys.argv[1];rt=Runtime(run,'local-author-'+number,swap_policy='growth');error=None
try:
 probe="import torch,importlib.util,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'triton_available':importlib.util.find_spec('triton') is not None}))"
 rt.execute('legacy-runtime-inspection',['docker','exec',C,'python3','-I','-c',probe],120)
 subprocess.run(['docker','start',M],check=True,timeout=30,stdout=subprocess.DEVNULL)
 rt.execute('local-qwen-author',['docker','exec',C,'python3','-I','/workspace/acceptance/request_nvfp4_candidate.py',run.name,number],900)
 (run/f'task-{number}-author.log').write_bytes((run/'local-qwen-author.log').read_bytes())
except BaseException as e:error=e;raise
finally:rt.close(error)
