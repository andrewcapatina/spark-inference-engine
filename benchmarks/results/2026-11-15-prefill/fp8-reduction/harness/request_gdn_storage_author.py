"""Guarded local-Qwen code authorship; no engine code synthesized here."""
from pathlib import Path
import json,subprocess,sys,hashlib
from nvfp4_update_runtime import Runtime,R,P,C,M
run=Path((R/'feedback/gdn-storage-run.txt').read_text().strip());number=sys.argv[1];rt=Runtime(run,'local-author-'+number,swap_policy='growth');error=None
try:
 subprocess.run(['docker','start',M],check=True,timeout=30,stdout=subprocess.DEVNULL)
 rt.execute('local-qwen-author',['docker','exec',C,'python3','-I','/workspace/acceptance/request_nvfp4_candidate.py',run.name,number],900)
 (run/('task-'+number+'-author.log')).write_bytes((run/'local-qwen-author.log').read_bytes())
except BaseException as e:
 error=e;raise
finally:
 rt.close(error)
