"""Fixed eager FP32 state-expression oracle; local-model kernel alone is the candidate."""
from pathlib import Path
import sys,os,json,gc,time,importlib.util,statistics
sys.path.insert(0,'/workspace')
def main():
 import torch
 import spark_engine.qwen_gdn_state_tail as tail
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache');torch.set_num_threads(4);torch.manual_seed(41041);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'running','mode':mode,'scope':'Isolated pointwise state tail, not GDN integration or checkpoint speedup; original FP32 multiply then add is fixed reference','cases':[],'gradient_cases':0,'chain_cases':[],'timings':[],'first_use_records':[]};seen=set();launches=[0];original_kernel=tail._state_tail_kernel
 class ObservedKernel:
  def __getitem__(self,grid):
   fn=original_kernel[grid]
   def call(*args,**kwargs):launches[0]+=1;return fn(*args,**kwargs)
   return call
 tail._state_tail_kernel=ObservedKernel()
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def reference(s,decay,update):return s*decay.unsqueeze(-1).unsqueeze(-1)+update
 def bits(a,b):
  assert a.shape==b.shape and a.dtype==b.dtype
  dt=torch.int16 if a.dtype==torch.bfloat16 else torch.int64 if a.dtype==torch.float64 else torch.int32
  assert torch.equal(a.contiguous().view(dt),b.contiguous().view(dt)),'FP32 rounding/ownership changed'
 def check(s,decay,u,eligible,label):
  snapshots=[v.clone() for v in [s,decay,u]];expected=reference(s,decay,u);launches[0]=0;torch.cuda.synchronize();start=time.perf_counter();actual=tail.state_tail(s,decay,u);torch.cuda.synchronize();elapsed=time.perf_counter()-start;bits(actual,expected);assert launches[0]==int(eligible and s.numel()>0),label
  assert not s.numel() or (actual.data_ptr()!=s.data_ptr() and actual.data_ptr()!=u.data_ptr());assert torch.isfinite(actual).all()
  for a,b in zip([s,decay,u],snapshots):bits(a,b)
  if eligible and s.shape[-2: ] not in seen:d['first_use_records'].append({'shape':list(s.shape),'first_use_seconds':elapsed});seen.add(s.shape[-2:])
  d['cases'].append({'shape':list(s.shape),'label':label,'fast_launches':launches[0],'bitwise_output_and_input_ownership':True});save()
 save()
 with torch.inference_mode():
  for b,h in [(1,1),(2,3),(1,48),(2,48)]:
   for k,v in [(1,1),(3,5),(64,128),(128,128),(128,64),(127,129)]:
    for variant in ['random','zero_state','zero_decay','unit_decay','zero_update']:
     s=torch.randn(b,h,k,v,device='cuda');decay=torch.rand(b,h,device='cuda');u=torch.randn_like(s)
     if variant=='zero_state':s.zero_()
     elif variant=='zero_decay':decay.zero_()
     elif variant=='unit_decay':decay.fill_(1)
     elif variant=='zero_update':u.zero_()
     check(s,decay,u,True,variant)
  edge=torch.tensor([0.,-0.,1.401298464324817e-45,-1.401298464324817e-45,1.1754942106924411e-38,-1.1754942106924411e-38,1.1754943508222875e-38,-1.1754943508222875e-38,3.4028234663852886e38,-3.4028234663852886e38],device='cuda').reshape(1,1,1,10)
  for scale in [0.,.5,1.]:check(edge,torch.full((1,1),scale,device='cuda'),torch.zeros_like(edge),True,'signedzero_subnormal_finite_extremes')
  s=torch.full((1,1,1,128),1+2**-23,device='cuda');decay=torch.full((1,1),1-2**-23,device='cuda');u=torch.full_like(s,-1);assert torch.count_nonzero(reference(s,decay,u))==0;check(s,decay,u,True,'must_not_fuse_multiply_add')
  for shape in [(0,48,128,128),(1,0,128,128)]:check(torch.empty(shape,device='cuda'),torch.empty(shape[:2],device='cuda'),torch.empty(shape,device='cuda'),True,'empty')
  for dtype in [torch.bfloat16,torch.float64]:
   s=torch.randn(2,3,16,16,device='cuda',dtype=dtype);decay=torch.rand(2,3,device='cuda',dtype=dtype);u=torch.randn_like(s);check(s,decay,u,False,'other_dtype_fallback')
  for dtype in [torch.float32,torch.bfloat16]:
   s=torch.randn(2,3,16,16,dtype=dtype);decay=torch.rand(2,3,dtype=dtype);u=torch.randn_like(s);check(s,decay,u,False,'cpu_fallback')
  s=torch.randn(2,3,16,16,device='cuda');decay=torch.rand(2,3,device='cuda');u=torch.randn_like(s);strided=torch.randn(2,3,16,32,device='cuda')[...,:16];check(strided,decay,u,False,'strided_state');check(s,decay,strided,False,'strided_update');check(s,torch.rand(2,6,device='cuda')[:,::2],u,False,'strided_decay');check(s,decay[:1],u,False,'broadcast_decay_fallback');del strided
  parameter=torch.nn.Parameter(s.clone(),requires_grad=True);check(parameter,decay,u,True,'parameter_in_inference');del parameter
  for steps in [1,16,128,768]:
   expected=torch.randn(1,48,128,128,device='cuda');actual=expected.clone();snapshots=[]
   for i in range(steps):
    decay=torch.rand(1,48,device='cuda');u=torch.randn_like(expected)*.01
    if i in [0,steps-1]:snapshots.append((actual,actual.clone()))
    expected=reference(expected,decay,u);actual=tail.state_tail(actual,decay,u);bits(actual,expected)
   for a,before in snapshots:bits(a,before)
   d['chain_cases'].append({'steps':steps,'every_transition_bitwise_exact':True,'old_state_snapshots_unchanged':True});del expected,actual,snapshots
  del edge,s,decay,u
 with torch.no_grad():
  s=torch.randn(1,3,16,16,device='cuda');decay=torch.rand(1,3,device='cuda');u=torch.randn_like(s);check(s,decay,u,False,'no_grad_fallback')
 with torch.enable_grad():
  for dtype in [torch.float32,torch.bfloat16]:
   args=[torch.randn(1,3,16,16,device='cuda',dtype=dtype,requires_grad=True),torch.rand(1,3,device='cuda',dtype=dtype,requires_grad=True),torch.randn(1,3,16,16,device='cuda',dtype=dtype,requires_grad=True)];other=[a.detach().clone().requires_grad_() for a in args];a=reference(*args);launches[0]=0;b=tail.state_tail(*other);assert launches[0]==0;bits(a,b);a.float().square().sum().backward();b.float().square().sum().backward()
   for x,y in zip(args,other):bits(x.grad,y.grad)
   d['gradient_cases']+=1;del args,other,a,b
 tail._state_tail_kernel=original_kernel
 with torch.inference_mode():
  s=torch.randn(1,48,128,128,device='cuda');decay=torch.rand(1,48,device='cuda');u=torch.randn_like(s)
  for fn in [reference,tail.state_tail]:fn(s,decay,u)
  for label in ['original','candidate','candidate','original']*2:
   fn=reference if label=='original' else tail.state_tail;a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(32):result=fn(s,decay,u)
   b.record();b.synchronize();d['timings'].append({'backend':label,'cuda_ms_per_call':a.elapsed_time(b)/32});del result
  d['median_cuda_ms_per_call']={label:statistics.median(x['cuda_ms_per_call'] for x in d['timings'] if x['backend']==label) for label in ['original','candidate']};del s,decay,u,a,b
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['cases'])==136 and d['gradient_cases']==2;d['status']='complete';save();print('STATE_TAIL_EXACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
