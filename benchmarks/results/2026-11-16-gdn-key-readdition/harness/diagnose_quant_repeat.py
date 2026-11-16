"""Isolate fixed-input projection and argmax repeatability; no whole-model load."""
from pathlib import Path
import sys,json,gc,importlib.util,inspect
sys.path.insert(0,'/workspace')
import torch
from safetensors import safe_open
from spark_engine.qwen_fp8 import FP8Linear
from spark_engine.qwen_nvfp4 import NVFP4Linear
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];d={'status':'running','mode':mode,'scope':'Fixed-input real-weight linear primitives and argmax, candidate absent; not whole-model validation','linear_cases':[],'argmax_cases':[]}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def get(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
with torch.inference_mode():
 for label,prefix in [('bf16_gate','model.language_model.layers.0.linear_attn.in_proj_a'),('fp8_head','lm_head'),('nvfp4_mlp','model.language_model.layers.0.mlp.gate_proj')]:
  if label=='bf16_gate':
   w=get(prefix+'.weight');module=None;fn=lambda x:torch.nn.functional.linear(x,w)
  elif label=='fp8_head':
   module=FP8Linear(get(prefix+'.weight'),get(prefix+'.weight_scale'));w=module.weight;fn=module.forward
  else:
   module=NVFP4Linear(get(prefix+'.weight_packed'),get(prefix+'.weight_scale'),get(prefix+'.weight_global_scale'),get(prefix+'.input_global_scale'));w=module.weight_packed;fn=module.forward
  width=w.shape[1]*(2 if label=='nvfp4_mlp' else 1)
  for rows in [1,16,256]:
   x=torch.randn(1,rows,width,device='cuda',dtype=torch.bfloat16);expected=fn(x).cpu();row={'primitive':label,'rows':rows,'input_width':width,'repeats':0,'differences':[]}
   repeats=512 if rows==1 else 32
   for i in range(repeats):
    if i%2:
     junk=[torch.full((n,),127 if i%4==1 else 165,device='cuda',dtype=torch.uint8) for n in [512,4096,65536,81920]];del junk
    actual=fn(x).cpu();row['repeats']+=1
    if not torch.equal(actual,expected):
     row['differences'].append({'iteration':i,'different_elements':int((actual!=expected).sum()),'max_abs':float((actual.float()-expected.float()).abs().nan_to_num().max()),'nonfinite':int((~torch.isfinite(actual)).sum())});torch.save({'input':x.cpu(),'expected':expected,'actual':actual},out/(label+'-first-difference.pt'));break
   d['linear_cases'].append(row);save();print(json.dumps(row),flush=True);del x,expected,actual
  del w,fn,module;gc.collect();torch.cuda.empty_cache()
 # Large-vocabulary argmax ties and output-allocation reuse, matching LM-head layout.
 base=torch.full((16,248320),-1.,device='cuda',dtype=torch.bfloat16);v=base[:1]
 for pair in [(9989,78326),(0,248319),(65535,131071)]:
  v.fill_(-1.);v[0,pair[0]]=32.;v[0,pair[1]]=32.;row={'tied_indices':list(pair),'expected_index':min(pair),'repeats':0,'wrong_indices':[]}
  for i in range(2048):
   actual=int(v.argmax(-1).item());row['repeats']+=1
   if actual!=min(pair):row['wrong_indices'].append({'iteration':i,'actual':actual});break
  d['argmax_cases'].append(row);save();print(json.dumps(row),flush=True)
 del base,v
gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('PROJECTION_ARGMAX_REPEATS_COMPLETE',flush=True)
