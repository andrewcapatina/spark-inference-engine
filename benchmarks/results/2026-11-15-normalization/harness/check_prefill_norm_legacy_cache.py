"""Small actual-kernel runtime preflight before loading another checkpoint."""
from pathlib import Path
import json
from nvfp4_update_runtime import Runtime,R,C
run=Path((R/'feedback/prefill-norm-main-run.txt').read_text().strip());rt=Runtime(run,'normalization-legacy-cache-preflight',swap_policy='growth');error=None
probe='''import torch,json
from pathlib import Path
from spark_engine.qwen_norm_tail import norm_tail
with torch.inference_mode():
 x=torch.linspace(-2,2,16,device='cuda',dtype=torch.float32).reshape(1,1,16);inv=torch.ones(1,1,1,device='cuda');w=torch.ones(16,device='cuda',dtype=torch.bfloat16);out=norm_tail(x,inv,w);expected=(x*inv*(w.float()+1)).bfloat16();torch.cuda.synchronize();assert torch.equal(out.view(torch.int16),expected.view(torch.int16))
print(json.dumps({'status':'passed','torch':torch.__version__,'cache':__import__('os').environ['TRITON_CACHE_DIR'],'mounts':[x for x in Path('/proc/mounts').read_text().splitlines() if ' /tmp ' in x]}))
'''
try:rt.execute('legacy-cache-small-kernel',['docker','exec','-e','TRITON_CACHE_DIR=/workspace/benchmark_runs/prefill-normalization-20261115T165203Z/triton-legacy-cache',C,'python3','-c',probe],120)
except BaseException as e:error=e;raise
finally:rt.close(error)
