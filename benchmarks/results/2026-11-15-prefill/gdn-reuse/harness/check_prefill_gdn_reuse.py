"""Exact synthetic GDN state tests and tiny-model integration, not checkpoint validation."""
from pathlib import Path
import sys,json,gc,tempfile,textwrap,importlib.util,statistics
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_ops as ops
 import spark_engine.qwen_layers as layers
 import test_quantized_loader as fixture
 from spark_engine.qwen_quantized import load_quantized_qwen
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(28028);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 source=Path('/harness/gated_delta_chunk_before_prefill_reuse.py.txt').read_text();ns=dict(vars(ops));exec(compile(textwrap.dedent(source),'retained-gdn-chunk-before-reuse','exec'),ns);original=ns['gated_delta_chunk'];candidate=ops.gated_delta_chunk;d={'status':'running','mode':mode,'scope':'Exact synthetic/real-layout and tiny-model tests only; isolated timings are not checkpoint speedups','direct_cases':[],'block_cases':[],'isolated_layout_timings':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def equal(a,b):
  if torch.is_tensor(a):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b),'Tensor changed'
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
  previous={name:getattr(torch,name) for name in ['tril','triu']};counts=[0]
  def wrapper(name):
   def call(*a,**kw):counts[0]+=1;return previous[name](*a,**kw)
   return call
  for name in previous:setattr(torch,name,wrapper(name))
  try:result=fn(*args,initial_state=state,chunk_size=chunk)
  finally:
   for name,old in previous.items():setattr(torch,name,old)
  return result,counts[0]
 def direct(batch,length,heads,kdim,vdim,dtype,has_state,chunk=64):
  args=inputs(batch,length,heads,kdim,vdim,dtype);state=torch.randn(batch,heads,kdim,vdim,device='cuda',dtype=torch.float32)*.01 if has_state else None;snapshot=[x.clone() for x in args];before=state.clone() if state is not None else None
  a,old_count=counted(original,args,state,chunk);b,new_count=counted(candidate,args,state,chunk);equal(a,b);equal(list(args),snapshot)
  if state is not None:equal(state,before)
  assert old_count==(length+chunk-1)//chunk and new_count==1,'Mask construction count changed unexpectedly'
  next_args=inputs(batch,3,heads,kdim,vdim,dtype);equal(ops.gated_delta_rule(*next_args,initial_state=a[1]),ops.gated_delta_rule(*next_args,initial_state=b[1]))
  d['direct_cases'].append({'batch':batch,'length':length,'heads':heads,'key_dim':kdim,'value_dim':vdim,'dtype':str(dtype),'initial_state':has_state,'chunk_size':chunk,'exact_output_state_inputs_and_continuation':True,'mask_builds_before':old_count,'mask_builds_after':new_count})
 def block_case(block,kind,batch,length):
  hidden=torch.randn(batch,length,256,device='cuda',dtype=torch.bfloat16)[...,::2];before=hidden.clone();ops.gated_delta_chunk=original;a,ca=block.prefill(hidden);ops.gated_delta_chunk=candidate;b,cb=block.prefill(hidden);equal((a,ca),(b,cb));equal(hidden,before);next_hidden=torch.randn(batch,1,128,device='cuda',dtype=torch.bfloat16);equal(block.decode(next_hidden,ca),block.decode(next_hidden,cb));d['block_cases'].append({'weights':kind,'batch':batch,'length':length,'exact_prefill_cache_continuation':True})
 save()
 try:
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
   for length in [256,1024]:
    args=inputs(1,length,48,128,128,torch.bfloat16)
    for fn in [original,candidate]:
     for _ in range(3):result=fn(*args);del result
    times=[]
    for backend in ['original','reuse','reuse','original']*2:
     fn=original if backend=='original' else candidate;a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();result=fn(*args);b.record();b.synchronize();times.append({'backend':backend,'cuda_ms':a.elapsed_time(b)});del result
    medians={backend:statistics.median(x['cuda_ms'] for x in times if x['backend']==backend) for backend in ['original','reuse']};d['isolated_layout_timings'].append({'length':length,'synthetic_real_head_layout':True,'samples':times,'medians':medians,'speedup':medians['original']/medians['reuse']});del args
 finally:ops.gated_delta_chunk=candidate
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['direct_cases'])==298 and len(d['block_cases'])==32;d['status']='complete';save();print('PREFILL_GDN_EXACT_PASSED direct='+str(len(d['direct_cases']))+' blocks='+str(len(d['block_cases'])),flush=True)
if __name__=='__main__':main()
