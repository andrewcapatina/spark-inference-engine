"""Isolate fixed-input projection and argmax repeatability; no whole-model load."""
from pathlib import Path
import sys,json,gc,importlib.util,inspect
sys.path.insert(0,'/workspace')
import torch
from safetensors import safe_open
from spark_engine.qwen_fp8 import FP8Linear
from spark_engine.qwen_nvfp4 import NVFP4Linear
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];d={'status':'running','mode':mode,'scope':'FP8 LM-head initialized output sentinel diagnostic, candidate absent; not whole-model validation','linear_cases':[],'argmax_cases':[]}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def get(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
with torch.inference_mode():
 module=FP8Linear(get('lm_head.weight'),get('lm_head.weight_scale'));x=torch.randn(1,1,5120,device='cuda',dtype=torch.bfloat16)
 expected=module(x).cpu();x2=x.reshape(-1,5120);sa=torch.clamp(x2.abs().amax(-1,keepdim=True).float()/448.,min=1e-6);a=(x2/sa).to(torch.float8_e4m3fn);a=torch.nn.functional.pad(a,(0,0,0,15));sa=torch.nn.functional.pad(sa,(0,0,0,15));b=module.weight.t();sb=module.weight_scale.t();d['sentinel_cases']=[]
 for sentinel in [float('nan'),123.,-123.]:
  dest=torch.full((16,248320),sentinel,device='cuda',dtype=torch.bfloat16)
  result=torch.ops.aten._scaled_mm.out(a,b,sa,sb,out_dtype=torch.bfloat16,use_fast_accum=True,out=dest)
  actual=result[:1].reshape(1,1,-1).cpu();allrows=result.cpu();row={'sentinel':str(sentinel),'output_aliases_buffer':result.data_ptr()==dest.data_ptr(),'valid_row_matches_reference':torch.equal(actual,expected),'valid_row_nonfinite':int((~torch.isfinite(actual)).sum()),'all_rows_nonfinite':int((~torch.isfinite(allrows)).sum()),'residual_sentinel_allrows':int((allrows==sentinel).sum()) if sentinel==sentinel else int(torch.isnan(allrows).sum())};d['sentinel_cases'].append(row);print(json.dumps(row),flush=True);save();del dest,result,actual,allrows
 del module,x,expected,x2,sa,a,b,sb

gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('PROJECTION_ARGMAX_REPEATS_COMPLETE',flush=True)
