"""Same matrices, strict bitwise gates and output layouts for local-model solver variants."""
from pathlib import Path
import sys,json,gc,statistics,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 import spark_engine.qwen_ops as ops
 from spark_engine.qwen_layers import QwenGatedDeltaNet
 from spark_engine.qwen_gdn_solve_variants import solve_variant
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(52052);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);variants=['columns','columns_fortran','batch','batch_fortran'];d={'status':'running','mode':mode,'scope':'Solver layout diagnostic only; all numerical cases must be bitwise exact per variant before integration. No downloaded checkpoint loaded.','cases':[],'timings':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def bit_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.int64 if a.dtype==torch.float64 else torch.int32),b.contiguous().view(torch.int64 if b.dtype==torch.float64 else torch.int32))
 def reference(a,b,c):return torch.linalg.solve_triangular(a,b,upper=False,unitriangular=True),torch.linalg.solve_triangular(a,c,upper=False,unitriangular=True)
 def examine(a,b,c,label):
  expected=reference(a,b,c);snapshots=[v.clone() for v in [a,b,c]]
  for variant in variants:
   actual=solve_variant(a,b,c,variant);assert isinstance(actual,tuple) and len(actual)==2;exact=[]
   for x,y in zip(expected,actual):assert x.shape==y.shape and x.dtype==y.dtype and torch.isfinite(y).all();exact.append({'bitwise_exact':bit_equal(x,y),'max_abs':(x-y).abs().max().item(),'different_elements':int(torch.count_nonzero(x!=y)),'expected_stride':list(x.stride()),'actual_stride':list(y.stride())})
   for x,y in zip([a,b,c],snapshots):assert bit_equal(x,y),'Input mutated'
   d['cases'].append({'label':label,'variant':variant,'system_shape':list(a.shape),'rhs_widths':[b.shape[-1],c.shape[-1]],'input_rhs_strides':[list(b.stride()),list(c.stride())],'outputs':exact,'all_outputs_bitwise_exact':all(x['bitwise_exact'] for x in exact)});save()
 save()
 with torch.inference_mode():
  for batch,heads,n,nv,nk in [(1,1,16,16,16),(1,3,64,16,32),(2,2,64,128,128),(1,48,64,128,128),(1,48,64,128,64),(1,2,64,127,129),(1,1,32,128,128)]:
   for variant in ['random','identity','zero_v','zero_k']:
    for layout in ['contiguous','gapped','transposed']:
     a=torch.tril(torch.randn(batch,heads,n,n,device='cuda')*.01)
     def rhs(width):
      if layout=='gapped':return (torch.randn(batch,heads,n,width*2,device='cuda')*.1)[...,::2]
      if layout=='transposed':return torch.randn(batch,heads,width,n,device='cuda').transpose(-1,-2)*.1
      return torch.randn(batch,heads,n,width,device='cuda')*.1
     b,c=rhs(nv),rhs(nk)
     if variant=='identity':a.zero_()
     elif variant=='zero_v':b.zero_()
     elif variant=='zero_k':c.zero_()
     examine(a,b,c,variant+'_'+layout)
  for dtype in [torch.float32,torch.float64]:
   a=torch.tril(torch.randn(1,2,16,16,dtype=dtype)*.01);b=torch.randn(1,2,16,16,dtype=dtype)*.1;c=torch.randn(1,2,16,32,dtype=dtype)*.1;examine(a,b,c,'cpu_diagnostic')
  ns={};exec(compile(Path('/harness/qwen_ops_before_solve_processing.py.txt').read_text(),'retained-original-gdn','exec'),ns);original=ns['gated_delta_chunk'];old=ops.gated_delta_chunk;original_solve=torch.linalg.solve_triangular;captured=[]
  def observe(a,b,*args,**kwargs):
   captured.append((a,b))
   return original_solve(a,b,*args,**kwargs)
  model=QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval();hidden=torch.randn(1,1024,128,device='cuda',dtype=torch.bfloat16);ops.gated_delta_chunk=original;torch.linalg.solve_triangular=observe
  try:output,cache=model.prefill(hidden)
  finally:ops.gated_delta_chunk=old;torch.linalg.solve_triangular=original_solve
  assert len(captured)==32
  for i in range(0,len(captured),2):
   ai,bi=captured[i];ci=captured[i+1][1];assert bit_equal(ai,captured[i+1][0]);examine(ai,bi,ci,'synthetic_full_block_chunk'+str(i//2))
  del ai,bi,ci
  a,b=captured[0];c=captured[1][1];d['representative_rhs_strides']=[list(b.stride()),list(c.stride())];del model,hidden,output,cache
  for variant in variants:
   fn=lambda aa,bb,cc:solve_variant(aa,bb,cc,variant)
   for method in [reference,fn]:
    for _ in range(3):result=method(a,b,c);del result
   for backend in ['original',variant,variant,'original']*2:
    method=reference if backend=='original' else fn;e1,e2=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e1.record()
    for _ in range(16):result=method(a,b,c)
    e2.record();e2.synchronize();d['timings'].append({'comparison':variant,'backend':backend,'cuda_ms_per_pair':e1.elapsed_time(e2)/16});del result
  d['variant_results']={v:{'all_cases_bitwise_exact':all(x['all_outputs_bitwise_exact'] for x in d['cases'] if x['variant']==v),'failing_cases':sum(not x['all_outputs_bitwise_exact'] for x in d['cases'] if x['variant']==v),'median_cuda_ms':{b:statistics.median(x['cuda_ms_per_pair'] for x in d['timings'] if x['comparison']==v and x['backend']==b) for b in ['original',v]}} for v in variants};del a,b,c,captured,e1,e2
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['cases'])==408;d['status']='complete';save();print(json.dumps(d['variant_results']),flush=True)
if __name__=='__main__':main()
