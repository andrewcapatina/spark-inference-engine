"""Reference arithmetic and headers only; no candidate kernel or whole model."""
from pathlib import Path
import sys,json,gc,collections,importlib.util
sys.path.insert(0,'/workspace')
import numpy as np
import torch
from safetensors import safe_open
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
with torch.inference_mode():
 bits=torch.arange(32640,device='cuda',dtype=torch.int32).to(torch.int16);m=bits.view(torch.bfloat16).reshape(-1,1);v=m.float();scalar=(v/448.0).clamp(min=1e-6);multiply=(v*(1.0/448.0)).clamp(min=1e-6);tensor=(v/torch.full_like(v,448.0)).clamp(min=1e-6);cpu=torch.from_numpy(np.maximum(v.cpu().numpy()/np.float32(448.0),np.float32(1e-6)))
 def compare(a,b):
  a,b=a.cpu(),b.cpu();diff=a.view(torch.int32)!=b.view(torch.int32);idx=diff.nonzero()[:8,0];return {'different':int(diff.sum()),'first_examples':[{'bf16_absmax':float(v[i].item()),'a':float(a[i].item()),'b':float(b[i].item())} for i in idx.tolist()]}
 comparisons={'scalar_div_vs_reciprocal_multiply':compare(scalar,multiply),'scalar_div_vs_tensor_div':compare(scalar,tensor),'scalar_div_vs_ieee_cpu_div':compare(scalar,cpu)}
 q=(m/scalar).float();q_cpu=torch.from_numpy(m.cpu().float().numpy()/scalar.cpu().numpy());comparisons['dynamic_tensor_div_vs_ieee_cpu_div']=compare(q,q_cpu)
 del bits,m,v,scalar,multiply,tensor,cpu,q,q_cpu
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];widths=collections.Counter();names=[]
for key,shard in index.items():
 if key.endswith('.weight') and key[:-7]+'.weight_scale' in index:
  with safe_open(str(root/shard),framework='pt',device='cpu') as f:shape=f.get_slice(key).get_shape()
  widths[shape[1]]+=1;names.append({'name':key,'shape':shape})
gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);released=bench.clear_cuda_library_workspaces(torch);memory=bench.memory(torch);assert memory['cuda_allocated_bytes']<=16*1024**2;d={'status':'complete','mode':mode,'scope':'Reference arithmetic/headers only; no model loaded or generation','torch':torch.__version__,'cuda':torch.version.cuda,'finite_nonnegative_bf16_values':32640,'comparisons':comparisons,'fp8_weight_count':len(names),'input_width_histogram':dict(widths),'weight_headers':names,'released_library_workspaces':released,'released_memory':memory};(out/'results.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ['weight_headers','released_memory']}),flush=True)
