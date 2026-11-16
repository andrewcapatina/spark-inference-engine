"""Explicit prefill dispatch; unchanged fallback/autograd/ordinary/decode behavior."""
from pathlib import Path
import sys,json,gc,tempfile,builtins,importlib.util
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_ops as ops
 import spark_engine.qwen_layers as layers
 import spark_engine.qwen_gated_norm_tail as tail
 import test_quantized_loader as fixture
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(38038);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 ns={};exec(compile(Path('/harness/qwen_ops_before_gated_processing.py.txt').read_text(),'retained-original-gated-ops','exec'),ns);original=ns['gated_rms_norm'];candidate=ops.gated_rms_norm_prefill;actual=tail.gated_norm_tail;calls=[0];d={'status':'running','mode':mode,'scope':'Primitive/gradient/tiny-model dispatch tests only, not real-checkpoint validation','primitive_cases':[],'model_cases':[],'gradient_cases':0}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def bits(a,b):
  assert a.dtype==b.dtype and a.shape==b.shape
  dt=torch.int16 if a.dtype in (torch.bfloat16,torch.float16) else torch.int32
  assert torch.equal(a.contiguous().view(dt),b.contiguous().view(dt)),'Bitwise normalization changed'
 def equal(a,b):
  if torch.is_tensor(a):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:equal(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):equal(x,y)
  else:assert a==b
 def observed(x,inv,w,gate):calls[0]+=1;assert x.device==inv.device==w.device==gate.device;return actual(x,inv,w,gate)
 def primitive(x,w,z,expected_calls,label):
  snapshots=[v.clone() for v in [x,w,z]];a=original(x,w,z);calls[0]=0;b=candidate(x,w,z);assert calls[0]==expected_calls,label;bits(a,b)
  for v,snapshot in zip([x,w,z],snapshots):bits(v,snapshot)
  d['primitive_cases'].append({'label':label,'shape':list(x.shape),'dtype':str(x.dtype),'weight_dtype':str(w.dtype),'tail_calls':calls[0],'exact':True});save()
 def model_case(model,batch,length):
  ids=torch.randint(0,128,(batch,length),device='cuda');expected=None
  for label,fn in [('original',original),('candidate',candidate)]:
   ops.gated_rms_norm_prefill=fn;calls[0]=0;logits,cache=model.prefill(ids);count=calls[0]
   if label=='candidate':assert (count>0)==(length>=16)
   first=fixture.cpu((logits,cache));calls[0]=0;next_logits,next_cache=model.decode(logits[:,-1,:].argmax(-1,keepdim=True),cache);assert calls[0]==0;second=fixture.cpu((next_logits,next_cache));calls[0]=0;full=fixture.cpu(model(ids));assert calls[0]==0;generated=model.generate(ids,16,None).cpu();values=(first,second,full,generated)
   if expected is None:expected=values
   else:equal(values,expected)
   del logits,cache,next_logits,next_cache
  d['model_cases'].append({'batch':batch,'length':length,'prefill_forward_nextdecode_cache_16ids_exact':True,'decode_and_forward_tail_calls':0});save()
 save();tail.gated_norm_tail=observed
 try:
  with torch.inference_mode():
   for width in [16,128,256]:
    for shape in [(1,width),(15,width),(16,width),(17,width),(48,width),(65,width),(768,width),(12288,width),(1,17,width),(2,65,width)]:
     x=torch.randn(shape,device='cuda',dtype=torch.bfloat16);w=torch.randn(width,device='cuda',dtype=torch.bfloat16)*.1;z=torch.randn_like(x);primitive(x,w,z,1,'eligible_explicit_prefill')
   for dtype in [torch.float32,torch.float16]:
    x=torch.randn(65,128,device='cuda',dtype=dtype);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);z=torch.randn_like(x);primitive(x,w,z,0,'non_bf16_fallback')
   for dtype in [torch.bfloat16,torch.float32]:
    x=torch.randn(65,128,dtype=dtype);w=torch.randn(128,dtype=dtype);z=torch.randn_like(x);primitive(x,w,z,0,'cpu_fallback')
   x=torch.randn(65,128,device='cuda',dtype=torch.bfloat16);z=torch.randn_like(x)
   for w in [torch.randn(128,device='cuda'),torch.randn(256,device='cuda',dtype=torch.bfloat16)[::2],torch.randn(1,128,device='cuda',dtype=torch.bfloat16)]:primitive(x,w,z,0,'unsupported_weight_fallback')
   w=torch.randn(128,device='cuda',dtype=torch.bfloat16);primitive(x,w,z.float(),0,'fp32_gate_input_fallback');strided=torch.randn(65,256,device='cuda',dtype=torch.bfloat16)[:,:128];primitive(strided,w,z,0,'noncontiguous_x_fallback');primitive(x,w,strided,0,'noncontiguous_gate_fallback');del strided
   norm=layers.QwenRMSNorm(128).cuda().bfloat16();norm.weight.copy_(w);calls[0]=0;bits(norm(x,z),original(x,w,z));assert calls[0]==0;calls[0]=0;bits(norm(x,z,prefill=True),original(x,w,z));assert calls[0]==1;del norm,x,z,w
  with torch.no_grad():
   x=torch.randn(65,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);z=torch.randn_like(x);primitive(x,w,z,0,'no_grad_fallback');del x,w,z
  with torch.enable_grad():
   for dtype in [torch.float32,torch.bfloat16]:
    x=torch.randn(17,128,device='cuda',dtype=dtype,requires_grad=True);w=torch.randn(128,device='cuda',dtype=dtype,requires_grad=True);z=torch.randn_like(x,requires_grad=True);x2=x.detach().clone().requires_grad_();w2=w.detach().clone().requires_grad_();z2=z.detach().clone().requires_grad_();a=original(x,w,z);calls[0]=0;b=candidate(x2,w2,z2);assert calls[0]==0;bits(a,b);a.float().square().sum().backward();b.float().square().sum().backward();bits(x.grad,x2.grad);bits(w.grad,w2.grad);bits(z.grad,z2.grad);d['gradient_cases']+=1;del x,w,z,x2,w2,z2,a,b
  with torch.inference_mode():
   x=torch.randn(65,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);z=torch.randn_like(x);expected=original(x,w,z);previous_import=builtins.__import__
   for missing,fallback in [('triton',True),('triton.language',True),('unrelated_dependency',False)]:
    def blocked(name,*args,**kwargs):
     if name=='spark_engine.qwen_gated_norm_tail':raise ModuleNotFoundError('simulated import mentioning triton',name=missing)
     return previous_import(name,*args,**kwargs)
    builtins.__import__=blocked;calls[0]=0
    try:
     if fallback:bits(candidate(x,w,z),expected)
     else:
      try:candidate(x,w,z)
      except ModuleNotFoundError as e:assert e.name==missing
      else:raise AssertionError('Unrelated import swallowed')
     assert calls[0]==0
    finally:builtins.__import__=previous_import
   for error in [RuntimeError('sentinel'),ModuleNotFoundError('helper runtime import',name='triton.runtime')]:
    def failed(*args):raise error
    tail.gated_norm_tail=failed
    try:candidate(x,w,z)
    except BaseException as e:assert e is error
    else:raise AssertionError('Helper error swallowed')
    finally:tail.gated_norm_tail=observed
    error.__traceback__=None
   del error,failed,x,w,z,expected
   with tempfile.TemporaryDirectory() as tmp:
    fixture.create_fixture(Path(tmp));gc.collect();torch.cuda.empty_cache();model=fixture.load_quantized_qwen(tmp)
    for batch in [1,2]:
     for length in [1,15,16,17,63,64,65,255,256,257]:model_case(model,batch,length)
    del model
 finally:ops.gated_rms_norm_prefill=candidate;tail.gated_norm_tail=actual
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['primitive_cases'])==41 and len(d['model_cases'])==20 and d['gradient_cases']==2;d['status']='complete';save();print('GATED_NORM_INTEGRATION_EXACT_PASSED',flush=True)
if __name__=='__main__':main()
