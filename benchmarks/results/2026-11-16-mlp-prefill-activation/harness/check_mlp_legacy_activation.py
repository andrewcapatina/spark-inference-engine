"""Exhaustive BF16 unary mapping on the original-model Torch/Triton runtime."""
import sys,gc,json
sys.path.insert(0,'/workspace')
import torch
from spark_engine.qwen_mlp_activation import silu_mul
torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.set_num_threads(4)
with torch.inference_mode():
 g=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
 for multiplier in [1.,-1.,0.,-0.,.5,3.25,3.3895313892515355e38,1.1754943508222875e-38,9.183549615799121e-41]:
  u=torch.full_like(g,multiplier);a=torch.nn.functional.silu(g)*u;b=silu_mul(g,u);same=(a.view(torch.int16)==b.view(torch.int16))|(torch.isnan(a)&torch.isnan(b));assert bool(same.all()),'Legacy-runtime BF16 rounding mismatch';del u,a,b
 del g
gc.collect();torch.cuda.empty_cache();assert torch.cuda.memory_allocated()<=16*1024**2;print(json.dumps({'status':'passed','torch':torch.__version__,'cuda':torch.version.cuda,'bf16_patterns':65536,'up_cases':9,'finite_bitwise_and_nan_mask_exact':True}),flush=True)
