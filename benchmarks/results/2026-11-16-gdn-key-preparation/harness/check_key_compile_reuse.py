"""Different full prompt lengths must reuse each layout's compiled key kernel."""
from pathlib import Path
import os,sys,json,time,gc
sys.path.insert(0,'/workspace')
import torch
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);cache=out/'fresh-triton-cache';os.environ['TRITON_CACHE_DIR']=str(cache)
from spark_engine.qwen_key_prepare import _make_key_preparer
torch.set_num_threads(4);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
d={'status':'running','mode':mode,'scope':'Cold-cache reuse and exact products across full prompt lengths; no model load','cases':[]}
with torch.inference_mode():
 for layout in ['time_major','head_major']:
  counts=[]
  for length in [128,256,512,1024]:
   k=torch.randn(1,length,48,128,device='cuda').transpose(1,2) if layout=='time_major' else torch.randn(1,48,length,128,device='cuda')
   beta=torch.rand(1,length,48,device='cuda').transpose(1,2) if layout=='time_major' else torch.rand(1,48,length,device='cuda')
   plan=_make_key_preparer(k,beta,64);exp=torch.rand(1,48,64,device='cuda');kk,bb=k[:,:,length-64:],beta[:,:,length-64:];torch.cuda.synchronize();start=time.perf_counter();actual=plan(kk,bb,exp);torch.cuda.synchronize();elapsed=time.perf_counter()-start
   expected=(kk*bb.unsqueeze(-1),(kk*bb.unsqueeze(-1))*exp.unsqueeze(-1))
   assert all(torch.equal(x.contiguous().view(torch.int32),y.contiguous().view(torch.int32)) for x,y in zip(actual,expected))
   count=len(list(cache.rglob('_key_prepare.json')));counts.append(count);d['cases'].append({'layout':layout,'full_prompt_length':length,'call_seconds':elapsed,'compiled_key_variants':count,'bitwise_exact':True});print(d['cases'][-1],flush=True)
   del k,beta,plan,exp,kk,bb,actual,expected
  assert len(set(counts))==1,'Prompt length caused recompilation within layout: '+str(counts)
gc.collect();torch.cuda.empty_cache();d['status']='complete';(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
