"""Black-box cache value, lifetime, allocation and autograd contract checks."""
from pathlib import Path
import sys,json,gc,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 from spark_engine.qwen_kv_cache import append_attention_cache as append
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True)
 torch.set_num_threads(4);torch.manual_seed(23023);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'running','mode':mode,'scope':'Synthetic standalone helper contracts only; not engine integration or real27B validation','cases':[],'gradient_cases':0,'storage_reuses':0}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def check(actual,expected):
  assert isinstance(actual,tuple) and len(actual)==2
  for a,b in zip(actual,expected):assert a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b),'Cache tensor mismatch'
 def rand(b,h,t,dtype):return torch.randn(b,h,t,32,device='cuda',dtype=dtype)[...,::2]
 def reference(cache,k,v):return torch.cat([cache[0],k],dim=2),torch.cat([cache[1],v],dim=2)
 def snapshot(cache):return tuple(t.clone() for t in cache)
 def unchanged(retained):
  for cache,values in retained:check(cache,values)
 with torch.inference_mode():
  for dtype in [torch.float32,torch.bfloat16]:
   for batch in [1,2]:
    for heads in [1,2,4]:
     for length in [0,1,63,64,65,255,256,257,511,512,513,1024]:
      initial=(rand(batch,heads,length,dtype),rand(batch,heads,length,dtype));cache=initial;expected=snapshot(initial);retained=[(cache,snapshot(cache))];versions=[];reuses=0
      for step in range(6):
       n=1 if step%2==0 else 3;k,v=rand(batch,heads,n,dtype),rand(batch,heads,n,dtype);inputs=(k.clone(),v.clone());old=cache
       expected=reference(expected,k,v);cache=append(cache,k,v);check(cache,expected);check((k,v),inputs);unchanged(retained)
       if step>0 and old[0].untyped_storage().nbytes()>=cache[0].numel()*cache[0].element_size():
        assert all(a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr() for a,b in zip(cache,old)),'Sequential append failed to reuse available backing storage';reuses+=1
       for t in cache:
        capacity=t.untyped_storage().nbytes()//(batch*heads*16*t.element_size());assert t.shape[2]<=capacity<t.shape[2]+256,'Unbounded capacity overhead'
       versions.append(cache);retained.append((cache,snapshot(cache)))
      ancestor=versions[0];branches=[]
      for branch in range(2):
       k,v=rand(batch,heads,3,dtype),rand(batch,heads,3,dtype);expected_branch=reference(ancestor,k,v);actual=append(ancestor,k,v);check(actual,expected_branch);branches.append((actual,expected_branch));retained.append((actual,snapshot(actual)));unchanged(retained)
      for branch in [0,1,0,1]:
       old,expected_branch=branches[branch];k,v=rand(batch,heads,1,dtype),rand(batch,heads,1,dtype);expected_branch=reference(expected_branch,k,v);actual=append(old,k,v);check(actual,expected_branch);branches[branch]=(actual,expected_branch);retained.append((actual,snapshot(actual)));unchanged(retained)
      d['cases'].append({'dtype':str(dtype),'batch':batch,'kv_heads':heads,'prefix_tokens':length,'six_sequential_appends_exact':True,'forks_and_interleaved_branches_exact':True,'bounded_capacity':True,'storage_reuses':reuses});d['storage_reuses']+=reuses
      del initial,cache,expected,retained,versions,branches,ancestor,old,actual,expected_branch,k,v,inputs
 with torch.enable_grad():
  for dtype in [torch.float32,torch.bfloat16]:
   for batch in [1,2]:
    past=tuple(torch.randn(batch,2,3,16,device='cuda',dtype=dtype,requires_grad=True) for _ in range(2));new=tuple(torch.randn(batch,2,1,16,device='cuda',dtype=dtype,requires_grad=True) for _ in range(2));result=append(past,*new);check(result,reference(past,*new));assert type(result) is tuple;sum(t.float().sum() for t in result).backward()
    for t in past+new:assert torch.equal(t.grad,torch.ones_like(t)),'Autograd changed'
    d['gradient_cases']+=1
 # no_grad is deliberately a fallback: only inference_mode enables shared storage.
 with torch.no_grad():
  past=tuple(rand(1,2,3,torch.float32) for _ in range(2));new=tuple(rand(1,2,1,torch.float32) for _ in range(2));result=append(past,*new);check(result,reference(past,*new));assert type(result) is tuple;d['no_grad_fallback_case']=True
 past=tuple(torch.randn(1,2,3,16,device='cuda',requires_grad=True) for _ in range(2));new=tuple(torch.randn(1,2,1,16,device='cuda',requires_grad=True) for _ in range(2))
 with torch.inference_mode():
  result=append(past,*new);check(result,reference(past,*new));assert type(result) is tuple;d['requires_grad_inference_fallback_case']=True
 # Dtype promotion must retain torch.cat semantics, even inside inference mode.
 with torch.inference_mode():
  past=tuple(rand(1,2,3,torch.bfloat16) for _ in range(2));new=tuple(rand(1,2,1,torch.float32) for _ in range(2));result=append(past,*new);check(result,reference(past,*new));assert type(result) is tuple;d['dtype_promotion_case']=True
 del past,new,result,t;gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2
 assert len(d['cases'])==144 and d['gradient_cases']==4 and d['storage_reuses']>0;d['status']='complete';save();print('PREALLOCATED_CACHE_CONTRACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
