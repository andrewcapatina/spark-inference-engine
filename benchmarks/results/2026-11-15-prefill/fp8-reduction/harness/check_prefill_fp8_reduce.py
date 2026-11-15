"""Exact scale/FP8-byte/output checks, finite BF16 patterns, tiny-model integration."""
from pathlib import Path
import sys,json,gc,textwrap,tempfile,importlib.util,statistics
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_fp8 as fp8
 import test_quantized_loader as fixture
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(29029);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 ns=dict(vars(fp8));exec(compile(textwrap.dedent(Path('/harness/fp8_forward_before_prefill_reduce.py.txt').read_text()),'retained-fp8-before-reduce','exec'),ns);original=ns['forward'];candidate=fp8.FP8Linear.forward;d={'status':'running','mode':mode,'scope':'Synthetic exact FP8 primitive/tiny-model tests only; no real-checkpoint validation','projection_cases':[],'model_cases':[],'isolated_timings':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def equal(a,b):
  if torch.is_tensor(a):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Tensor changed'
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for key in a:equal(a[key],b[key])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):equal(x,y)
  else:assert a==b
 def layer(n,k):return fp8.FP8Linear(torch.randn(n,k,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn),torch.linspace(.0001,.0002,n,device='cuda').view(n,1))
 def watched(fn,block,x):
  mm,amax=torch._scaled_mm,torch.amax;captures=[];max_types=[]
  def observe_max(a,*args,**kwargs):max_types.append(str(a.dtype));return amax(a,*args,**kwargs)
  def observe_mm(a,b,sa,sb,*args,**kwargs):captures.append((a.view(torch.uint8).clone(),sa.clone(),dict(kwargs)));return mm(a,b,sa,sb,*args,**kwargs)
  torch._scaled_mm=observe_mm;torch.amax=observe_max
  try:result=fn(block,x)
  finally:torch._scaled_mm=mm;torch.amax=amax
  assert len(captures)==1 and len(max_types)==1;return result,captures[0],max_types[0]
 def projection(block,x,label):
  before=x.clone();a,old,old_dtype=watched(original,block,x);b,new,new_dtype=watched(candidate,block,x);equal(a,b);equal(old[0],new[0]);equal(old[1],new[1]);assert old[2]==new[2];assert old_dtype=='torch.float32' and new_dtype=='torch.bfloat16';assert old[1].dtype==torch.float32;equal(x,before);assert torch.isfinite(a).all();d['projection_cases'].append({'kind':label,'input_shape':list(x.shape),'input_contiguous':x.is_contiguous(),'scale_dtype':str(new[1].dtype),'exact_scales_fp8_bytes_and_output':True})
 def model_case(model,batch,length):
  ids=torch.randint(0,128,(batch,length),device='cuda');expected=None
  for backend in ['original','reuse']:
   fp8.FP8Linear.forward=original if backend=='original' else candidate;logits,cache=model.prefill(ids);first=fixture.cpu((logits,cache));next_id=logits[:,-1,:].argmax(-1,keepdim=True);next_logits,next_cache=model.decode(next_id,cache);second=fixture.cpu((next_logits,next_cache));generated=model.generate(ids,16,None);values=generated.cpu()
   if expected is None:expected=(first,second,values)
   else:equal((first,second,values),expected)
   del logits,cache,next_id,next_logits,next_cache,generated
  d['model_cases'].append({'batch':batch,'prompt_tokens':length,'prefill_next_decode_cache_and16_generated_ids_exact':True})
 save()
 try:
  with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:
   for k in [128,256]:
    block=layer(128,k);weight_before=block.weight.view(torch.uint8).clone();scale_before=block.weight_scale.clone()
    for length in [1,3,15,16,17,63,64,65,255,256,257,1024]:
     for prefix in [(),(1,),(2,)]:
      for strided in [False,True]:
       shape=prefix+(length,k*(2 if strided else 1));x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
       if strided:x=x[...,::2]
       projection(block,x,'random_padding_and_layout')
    equal(block.weight.view(torch.uint8),weight_before);equal(block.weight_scale,scale_before);del block,weight_before,scale_before,x
   block=fp8.FP8Linear(torch.eye(128,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn),torch.full((128,1),.0001,device='cuda'));patterns=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16);finite=patterns[torch.isfinite(patterns)];assert finite.numel()==65280;x=finite.reshape(-1,128).cuda();projection(block,x,'all65280_finite_bf16_patterns');d['finite_bf16_patterns_covered']=finite.numel();del patterns,finite,x;del block;block=layer(128,128)
   x=torch.tensor([0.,1e-40,1e-8,1e-6,.000447,.000448,.000449,1.,448.,1e20],dtype=torch.bfloat16,device='cuda')[:,None].expand(-1,128);projection(block,x,'zero_subnormal_clamp_boundary_and_large');del x
   for scale in [0.,1e-8,1.]:
    x=torch.randn(256,device='cuda',dtype=torch.bfloat16)[::2]*scale;projection(block,x,'vector_input');del x
   del block;gc.collect();torch.cuda.empty_cache();fixture.create_fixture(Path(tmp));gc.collect();torch.cuda.empty_cache();model=fixture.load_quantized_qwen(tmp)
   for batch in [1,2]:
    for length in [1,15,16,17,63,64,65,255,256,257]:model_case(model,batch,length)
   del model;fp8.FP8Linear.forward=candidate;gc.collect();torch.cuda.empty_cache()
   import test_native_fp8
   test_native_fp8.run();fixture.run();d['unchanged_independent_fp8_and_quantized_loader_checks_passed']=True
   gc.collect();torch.cuda.empty_cache();block=layer(10240,5120)
   for length in [256,1024]:
    x=torch.randn(length,5120,device='cuda',dtype=torch.bfloat16)
    for fn in [original,candidate]:
     for _ in range(3):value=fn(block,x);del value
    samples=[]
    for backend in ['original','reuse','reuse','original']*2:
     fn=original if backend=='original' else candidate;a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();value=fn(block,x);b.record();b.synchronize();samples.append({'backend':backend,'cuda_ms':a.elapsed_time(b)});del value
    medians={backend:statistics.median(s['cuda_ms'] for s in samples if s['backend']==backend) for backend in ['original','reuse']};d['isolated_timings'].append({'input_tokens':length,'synthetic_real_projection_shape':[10240,5120],'samples':samples,'medians':medians,'speedup':medians['original']/medians['reuse']});del x
   del block
 finally:fp8.FP8Linear.forward=candidate
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['projection_cases'])==149 and len(d['model_cases'])==20;d['status']='complete';save();print('PREFILL_FP8_EXACT_PASSED projections='+str(len(d['projection_cases']))+' models='+str(len(d['model_cases'])),flush=True)
if __name__=='__main__':main()
