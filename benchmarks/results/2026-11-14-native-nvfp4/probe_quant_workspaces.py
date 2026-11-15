"""Tiny library-workspace diagnosis only; no model is loaded."""
import torch,gc,json
import flashinfer

def exercise():
 a=torch.randn(128,128,device='cuda');b=a@a
 x=a.to(torch.bfloat16);y=x@x;z=torch.bmm(x.unsqueeze(0),x.unsqueeze(0))
 scale=torch.tensor([3.25],device='cuda');q,s=flashinfer.fp4_quantize(x,scale)
 f=flashinfer.mm_fp4(q,q.T,s.view(torch.uint8),s.view(torch.uint8).T,alpha=scale.reciprocal().square(),out_dtype=torch.bfloat16,backend='cutlass')
 torch.cuda.synchronize()
torch.cuda.init();print(json.dumps({'phase':'fresh','allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}),flush=True)
exercise();gc.collect();torch.cuda.empty_cache();print(json.dumps({'phase':'after_tiny_ops_only','allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}),flush=True)
torch._C._cuda_clearCublasWorkspaces();gc.collect();torch.cuda.empty_cache();print(json.dumps({'phase':'after_cublas_workspace_clear','allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}),flush=True)

import importlib.util
spec=importlib.util.spec_from_file_location("bench","/workspace/tools/benchmark_qwen.py");bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
print(json.dumps({"phase":"clear_all_idle_library_workspaces","cleared":bench.clear_cuda_library_workspaces(torch),"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}),flush=True)
