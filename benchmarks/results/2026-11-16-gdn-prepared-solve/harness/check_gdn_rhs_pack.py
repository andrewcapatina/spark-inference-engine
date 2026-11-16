"""Exact original FP32 RHS products/row-major concatenation versus local-model pack kernel."""
from pathlib import Path
import sys,os,json,gc,time,statistics,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 from spark_engine.qwen_gdn_rhs_pack import pack_rhs
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache');torch.set_num_threads(4);torch.manual_seed(46046);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'running','mode':mode,'scope':'Standalone FP32 RHS preparation bitwise test; no GDN/model integration or whole-checkpoint speed claim','cases':[],'timings':[],'first_use_seconds':None,'invalid_cases':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def reference(v,k,b,e):return torch.cat((v*b.unsqueeze(-1),(k*b.unsqueeze(-1))*e.unsqueeze(-1)),dim=-1)
 def bits(a,b):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)),'FP32 RHS changed'
 def check(v,k,b,e,label):
  old=[x.clone() for x in [v,k,b,e]];expected=reference(v,k,b,e);torch.cuda.synchronize();start=time.perf_counter();actual=pack_rhs(v,k,b,e);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  if d['first_use_seconds'] is None:d['first_use_seconds']=elapsed
  bits(actual,expected);assert actual.is_contiguous() and actual.stride()==expected.stride() and torch.isfinite(actual).all();assert not actual.numel() or all(actual.data_ptr()!=x.data_ptr() for x in [v,k,b,e])
  for x,y in zip([v,k,b,e],old):bits(x,y)
  d['cases'].append({'label':label,'v_shape':list(v.shape),'k_shape':list(k.shape),'input_strides':[list(x.stride()) for x in [v,k,b,e]],'output_stride':list(actual.stride()),'bitwise_output_and_inputs_exact':True});save()
 save()
 with torch.inference_mode():
  for b,h in [(1,1),(2,3),(1,48)]:
   for c in [1,3,64,65]:
    for nv,nk in [(1,1),(16,32),(128,128),(127,129)]:
     for layout in ['contiguous','gdn','gapped','transposed','broadcast']:
      def source(width):
       if layout=='gdn':return torch.randn(b,c,h,width,device='cuda').transpose(1,2)
       if layout=='gapped':return torch.randn(b,h,c,width*2,device='cuda')[...,::2]
       if layout=='transposed':return torch.randn(b,h,width,c,device='cuda').transpose(-1,-2)
       if layout=='broadcast':return torch.randn(1,1,c,width,device='cuda').expand(b,h,c,width)
       return torch.randn(b,h,c,width,device='cuda')
      v,k=source(nv),source(nk);beta=torch.rand(b,c*2,h*2,device='cuda')[:,::2,::2].transpose(1,2);exp=torch.rand_like(beta)
      if layout=='broadcast':beta=torch.rand(1,1,c,device='cuda').expand(b,h,c);exp=torch.rand(b,h,1,device='cuda').expand(b,h,c)
      check(v,k,beta,exp,layout)
  edge=torch.tensor([0.,-0.,1.401298464324817e-45,-1.401298464324817e-45,1.1754942106924411e-38,-1.1754942106924411e-38,1.1754943508222875e-38,-1.1754943508222875e-38,3.4028234663852886e38,-3.4028234663852886e38],device='cuda').reshape(1,1,1,10)
  for sign in [.5,-.5]:check(edge,edge,torch.full((1,1,1),sign,device='cuda'),torch.full((1,1,1),.5,device='cuda'),'signedzero_subnormal_finite_extremes')
  v=torch.randn(1,3,64,128,device='cuda');k=torch.randn_like(v);beta=torch.rand(1,3,64,device='cuda');exp=torch.rand_like(beta);ordered=(k*beta.unsqueeze(-1))*exp.unsqueeze(-1);reassociated=k*(beta*exp).unsqueeze(-1);d['reassociation_counterexample_elements']=int(torch.count_nonzero(ordered!=reassociated));assert d['reassociation_counterexample_elements']>0;check(v,k,beta,exp,'must_keep_two_multiply_roundings');del ordered,reassociated,edge
  check(torch.empty(0,3,64,128,device='cuda'),torch.empty(0,3,64,128,device='cuda'),torch.empty(0,3,64,device='cuda'),torch.empty(0,3,64,device='cuda'),'empty_batch')
  del v,k,beta,exp
  v=torch.randn(1,64,48,128,device='cuda').transpose(1,2);k=torch.randn_like(v);beta=torch.rand(1,48,64,device='cuda');exp=torch.rand_like(beta)
  for fn in [reference,pack_rhs]:
   for _ in range(3):result=fn(v,k,beta,exp);del result
  for label in ['original','candidate','candidate','original']*2:
   fn=reference if label=='original' else pack_rhs;a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(32):result=fn(v,k,beta,exp)
   z.record();z.synchronize();d['timings'].append({'backend':label,'cuda_ms_per_pack':a.elapsed_time(z)/32});del result
  d['median_cuda_ms_per_pack']={label:statistics.median(x['cuda_ms_per_pack'] for x in d['timings'] if x['backend']==label) for label in ['original','candidate']};del v,k,beta,exp,a,z
 with torch.inference_mode():
  args=[torch.ones(1,1,1,16,device='cuda'),torch.ones(1,1,1,16,device='cuda'),torch.ones(1,1,1,device='cuda'),torch.ones(1,1,1,device='cuda')]
  for label,index,value in [('cpu',0,args[0].cpu()),('dtype',2,args[2].bfloat16()),('rank',1,args[1].squeeze(0)),('prefix',2,torch.ones(1,2,1,device='cuda'))]:
   trial=list(args);trial[index]=value
   try:pack_rhs(*trial)
   except AssertionError:d['invalid_cases'].append(label)
   else:raise AssertionError('Invalid metadata accepted: '+label)
  huge=torch.ones(1,device='cuda').expand(1,1,1,2**31)
  try:pack_rhs(huge,huge,args[2],args[3])
  except AssertionError:d['invalid_cases'].append('int32_limit_without_large_allocation')
  else:raise AssertionError('Huge output bounds guard missing')
  del trial,value,huge
 try:pack_rhs(*args)
 except AssertionError:d['outside_inference_mode_rejected']=True
 else:raise AssertionError('Non-inference helper accepted')
 del args
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['cases'])==244;d['status']='complete';save();print('RHS_PACK_BITWISE_EXACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
