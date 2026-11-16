"""Exact synthetic GDN state tests and tiny-model integration, not checkpoint validation."""
from pathlib import Path
import sys,json,gc,tempfile,textwrap,importlib.util,statistics,builtins
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_ops as ops
 import spark_engine.qwen_layers as layers
 import test_quantized_loader as fixture
 from spark_engine.qwen_quantized import load_quantized_qwen
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(28028);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 source=Path('/harness/qwen_ops_before_batch_processing.py.txt').read_text();ns=dict(vars(ops));exec(compile(textwrap.dedent(source),'retained-gdn-before-state-tail','exec'),ns);original=ns['gated_delta_chunk'];candidate=ops.gated_delta_chunk;d={'status':'running','mode':mode,'scope':'Exact synthetic/real-layout and tiny-model tests only; isolated timings are not checkpoint speedups','direct_cases':[],'block_cases':[],'isolated_layout_timings':[],'model_cases':[],'gradient_cases':0}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def equal(a,b):
  if torch.is_tensor(a):
   if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
    d['failure']={'shape_before':list(a.shape),'shape_after':list(b.shape),'dtype_before':str(a.dtype),'dtype_after':str(b.dtype),'max_abs':float((a.float()-b.float()).abs().max()),'different_elements':int((a!=b).sum())};save();raise AssertionError('Tensor changed: '+str(d['failure']))
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for key in a:equal(a[key],b[key])
  elif isinstance(a,(tuple,list)):
   assert len(a)==len(b)
   for x,y in zip(a,b):equal(x,y)
  else:assert a==b
 def inputs(batch,length,heads,kdim,vdim,dtype):
  def rand(dim):return torch.randn(batch,length,heads,dim*2,device='cuda',dtype=dtype)[...,::2]
  return rand(kdim),rand(kdim),rand(vdim),-torch.rand(batch,length,heads,device='cuda',dtype=dtype)*.1,torch.rand(batch,length,heads,device='cuda',dtype=dtype)
 def counted(fn,args,state,chunk):
  float_before=torch.Tensor.float;value_casts=[0]
  def observe_float(t,*a,**kw):
   if t.data_ptr()==args[2].data_ptr():value_casts[0]+=1
   return float_before(t,*a,**kw)
  torch.Tensor.float=observe_float
  previous={name:getattr(torch,name) for name in ['tril','triu']};counts=[0]
  def wrapper(name):
   def call(*a,**kw):counts[0]+=1;return previous[name](*a,**kw)
   return call
  for name in previous:setattr(torch,name,wrapper(name))
  try:result=fn(*args,initial_state=state,chunk_size=chunk)
  finally:
   for name,old in previous.items():setattr(torch,name,old)
   torch.Tensor.float=float_before
  eligible=args[0].dtype==torch.bfloat16 and args[2].dtype==torch.bfloat16 and args[2].is_cuda and torch.is_inference_mode_enabled()
  assert value_casts[0]==(0 if eligible else 1),('Unexpected value float calls',value_casts,eligible)
  assert result[1].dtype==torch.float32 and result[0].dtype==args[0].dtype
  return result,counts[0]
 def direct(batch,length,heads,kdim,vdim,dtype,has_state,chunk=64):
  d['current_direct_case']={'batch':batch,'length':length,'heads':heads,'kdim':kdim,'vdim':vdim,'dtype':str(dtype),'initial_state':has_state,'chunk_size':chunk}
  args=inputs(batch,length,heads,kdim,vdim,dtype);state=torch.randn(batch,heads,kdim,vdim,device='cuda',dtype=torch.float32)*.01 if has_state else None;snapshot=[x.clone() for x in args];before=state.clone() if state is not None else None
  a,old_count=counted(original,args,state,chunk);b,new_count=counted(candidate,args,state,chunk);equal(a,b);equal(list(args),snapshot)
  if state is not None:equal(state,before)
  assert old_count==(length+chunk-1)//chunk and 1<=new_count<=old_count,'Unexpected preparation count'
  next_args=inputs(batch,3,heads,kdim,vdim,dtype);equal(ops.gated_delta_rule(*next_args,initial_state=a[1]),ops.gated_delta_rule(*next_args,initial_state=b[1]))
  d['direct_cases'].append({'batch':batch,'length':length,'heads':heads,'key_dim':kdim,'value_dim':vdim,'dtype':str(dtype),'initial_state':has_state,'chunk_size':chunk,'exact_output_state_inputs_and_continuation':True,'mask_builds_before':old_count,'mask_builds_after':new_count})
 def block_case(block,kind,batch,length):
  hidden=torch.randn(batch,length,256,device='cuda',dtype=torch.bfloat16)[...,::2];before=hidden.clone();ops.gated_delta_chunk=original;a,ca=block.prefill(hidden);ops.gated_delta_chunk=candidate;b,cb=block.prefill(hidden);equal((a,ca),(b,cb));equal(hidden,before);next_hidden=torch.randn(batch,1,128,device='cuda',dtype=torch.bfloat16);equal(block.decode(next_hidden,ca),block.decode(next_hidden,cb));d['block_cases'].append({'weights':kind,'batch':batch,'length':length,'exact_prefill_cache_continuation':True})
 save()
 try:
  with torch.no_grad():
   args=inputs(1,5,2,16,16,torch.bfloat16);a,_=counted(original,args,None,64);b,_=counted(candidate,args,None,64);equal(a,b);d['no_grad_original_path_exact']=True;del args,a,b
  with torch.enable_grad():
   for dtype in [torch.float32,torch.bfloat16]:
    args=[v.detach().clone().requires_grad_() for v in inputs(1,3,2,16,16,dtype)];other=[v.detach().clone().requires_grad_() for v in args];a,_=counted(original,args,None,64);b,_=counted(candidate,other,None,64);equal(a,b);(a[0].float().square().sum()+a[1].square().sum()).backward();(b[0].float().square().sum()+b[1].square().sum()).backward()
    for x,y in zip(args,other):equal(x.grad,y.grad)
    d['gradient_cases']+=1;del args,other,a,b,x,y
  with torch.inference_mode():
   args=inputs(1,65,2,16,16,torch.bfloat16);state=torch.randn(1,2,16,16,device='cuda').transpose(-1,-2);before=state.clone();equal(original(*args,initial_state=state),candidate(*args,initial_state=state));equal(state,before);del args,state,before
   d['strided_initial_state_exact']=True
  with torch.inference_mode():
   for dtype in [torch.float32,torch.bfloat16,torch.float64]:
    args=tuple(x.cpu().to(dtype) for x in inputs(1,65,2,16,32,torch.float32))
    for state_dtype in [None,torch.bfloat16,torch.float64]:
     state=None if state_dtype is None else torch.randn(1,2,16,32,dtype=state_dtype)
     a,_=counted(original,args,state,64);b,_=counted(candidate,args,state,64);equal(a,b)
   for state_dtype in [torch.bfloat16,torch.float64]:
    args=inputs(1,65,2,16,32,torch.bfloat16);state=torch.randn(1,2,32,16,device='cuda',dtype=state_dtype).transpose(-1,-2);before=state.clone()
    a,_=counted(original,args,state,64);b,_=counted(candidate,args,state,64);equal(a,b);equal(state,before)
   del args,state,a,b,before
   d['cpu_and_provided_state_dtype_cases']=11
  with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:
   for dtype in [torch.float32,torch.bfloat16]:
    for batch in [1,2]:
     for heads,kdim,vdim in [(3,16,16),(2,16,32)]:
      for length in [1,3,4,5,63,64,65,127,128,129,255,256,257,511,512,513,1024]:
       for state in [False,True]:direct(batch,length,heads,kdim,vdim,dtype,state)
   for length in [1,63,64,65,256,257,1024]:
    for state in [False,True]:direct(1,length,48,128,128,torch.bfloat16,state)
   for length in [1,15,16,17,33,65]:
    for state in [False,True]:direct(1,length,2,16,16,torch.float32,state,16)
   for kind in ['dense','mixed_fp8_nvfp4']:
    if kind=='dense':block=layers.QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':1,'linear_num_value_heads':3,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval()
    else:fixture.create_fixture(Path(tmp));model=load_quantized_qwen(tmp);block=model.layers[1].linear_attn
    for batch in [1,2]:
     for length in [1,63,64,65,255,256,257,1024]:block_case(block,kind,batch,length)
    del block
    if kind!='dense':del model
    gc.collect();torch.cuda.empty_cache()
   block=layers.QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval()
   for batch in [1,2]:
    for length in [65,129,256,257,513,1024]:block_case(block,'native_48_head_layout',batch,length)
   del block
   for length in [129,257,1024]:
    for has_state in [False,True]:direct(1,length,48,128,128,torch.bfloat16,has_state)
   fixture.create_fixture(Path(tmp));model=load_quantized_qwen(tmp)
   for batch in [1,2]:
    for length in [1,15,16,17,63,64,65,255,256,257]:
     ids=torch.randint(0,128,(batch,length),device='cuda');expected=None
     for label,fn in [('original',original),('candidate',candidate)]:
      ops.gated_delta_chunk=fn;logits,cache=model.prefill(ids);first=fixture.cpu((logits,cache));next_logits,next_cache=model.decode(logits[:,-1,:].argmax(-1,keepdim=True),cache);second=fixture.cpu((next_logits,next_cache));full=fixture.cpu(model(ids));generated=model.generate(ids,16,None).cpu();values=(first,second,full,generated)
      if expected is None:expected=values
      else:equal(values,expected)
      del logits,cache,next_logits,next_cache
     d['model_cases'].append({'batch':batch,'length':length,'prefill_forward_nextdecode_cache_16ids_exact':True});del ids,values,expected
   del model
   for length in [256,1024]:
    args=inputs(1,length,48,128,128,torch.bfloat16)
    for fn in [original,candidate]:
     for _ in range(3):result=fn(*args);del result
    times=[]
    for backend in ['original','candidate','candidate','original']*2:
     fn=original if backend=='original' else candidate;a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();result=fn(*args);b.record();b.synchronize();times.append({'backend':backend,'cuda_ms':a.elapsed_time(b)});del result
    medians={backend:statistics.median(x['cuda_ms'] for x in times if x['backend']==backend) for backend in ['original','candidate']};d['isolated_layout_timings'].append({'length':length,'synthetic_real_head_layout':True,'samples':times,'medians':medians,'speedup':medians['original']/medians['candidate']});del args
 finally:ops.gated_delta_chunk=candidate
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['direct_cases'])==304 and len(d['block_cases'])==44 and len(d['model_cases'])==20 and d['gradient_cases']==2;d['status']='complete';save();print('GDN_BATCH_INTEGRATION_EXACT_PASSED direct='+str(len(d['direct_cases']))+' blocks='+str(len(d['block_cases'])),flush=True)
if __name__=='__main__':main()
