"""Guarded local normalization versus retained original, including fallback/autograd/error paths."""
from pathlib import Path
import sys,json,gc,tempfile,builtins,importlib.util
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_ops as ops
 import spark_engine.qwen_layers as layers
 import spark_engine.qwen_norm_tail as tail
 import test_quantized_loader as fixture
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(35035);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 ns={};exec(compile(Path('/harness/qwen_ops_before_norm_processing.py.txt').read_text(),'retained-qwen-before-norm','exec'),ns);original=ns['qwen_rms_norm'];candidate=ops.qwen_rms_norm;actual_tail=tail.norm_tail;calls=[0];d={'status':'running','mode':mode,'scope':'Guarded primitive and tiny-model tests only; not real-checkpoint validation','primitive_cases':[],'model_cases':[],'gradient_cases':0}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def bits(a,b):
  assert a.dtype==b.dtype and a.shape==b.shape
  dtype=torch.int16 if a.dtype in [torch.bfloat16,torch.float16] else torch.int32
  assert torch.equal(a.contiguous().view(dtype),b.contiguous().view(dtype)),'Bitwise norm changed'
 def equal(a,b):
  if torch.is_tensor(a):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:equal(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):equal(x,y)
  else:assert a==b
 def observed(x,inv,w):calls[0]+=1;assert x.device==inv.device==w.device;return actual_tail(x,inv,w)
 def primitive(x,w,expected_calls,label):
  before=x.clone();wb=w.clone();a=original(x,w);calls[0]=0;b=candidate(x,w);assert calls[0]==expected_calls,label;bits(a,b);bits(x,before);bits(w,wb);d['primitive_cases'].append({'label':label,'shape':list(x.shape),'dtype':str(x.dtype),'weight_dtype':str(w.dtype),'tail_calls':calls[0],'exact':True});save()
 def select(fn):ops.qwen_rms_norm=fn;layers.qwen_rms_norm=fn
 def model_case(model,batch,length):
  ids=torch.randint(0,128,(batch,length),device='cuda');expected=None
  for backend,fn in [('original',original),('candidate',candidate)]:
   select(fn);calls[0]=0;logits,cache=model.prefill(ids);prefill_calls=calls[0]
   if backend=='candidate':assert (prefill_calls>0)==(length>=16)
   first=fixture.cpu((logits,cache));calls[0]=0;next_logits,next_cache=model.decode(logits[:,-1,:].argmax(-1,keepdim=True),cache);assert calls[0]==0;second=fixture.cpu((next_logits,next_cache));full=fixture.cpu(model(ids));generated=model.generate(ids,16,None).cpu();values=(first,second,full,generated)
   if expected is None:expected=values
   else:equal(values,expected)
   del logits,cache,next_logits,next_cache
  d['model_cases'].append({'batch':batch,'length':length,'prefill_forward_nextdecode_cache_16ids_exact':True,'decode_tail_calls':0});save()
 save();tail.norm_tail=observed
 try:
  with torch.inference_mode():
   for width in [16,128,256,5120]:
    for batch in [1,2]:
     for length in [1,15,16,17,63,64,65,256,1024]:
      x=torch.randn(batch,length,width,device='cuda',dtype=torch.bfloat16);w=torch.randn(width,device='cuda',dtype=torch.bfloat16)*.1;primitive(x,w,int(length>=16),'prefill_threshold_or_decode')
   for shape in [(1,1,24,256),(1,16,24,256),(1,65,4,256),(1,1024,24,256)]:
    x=torch.randn(*shape[:-1],shape[-1]*2,device='cuda',dtype=torch.bfloat16)[...,:shape[-1]];w=torch.randn(shape[-1],device='cuda',dtype=torch.float32)*.1;primitive(x,w,int(shape[1]>=16),'gapped_attention_shape')
   x=torch.randn(1,128,32,device='cuda',dtype=torch.bfloat16).transpose(1,2);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);primitive(x,w,0,'transposed_fp32_conversion_layout')
   for dtype in [torch.float32,torch.float16]:
    x=torch.randn(1,65,128,device='cuda',dtype=dtype);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);primitive(x,w,0,'non_bf16_fallback')
   for dtype in [torch.float32,torch.bfloat16]:
    x=torch.randn(1,65,128,dtype=dtype);w=torch.randn(128,dtype=dtype);primitive(x,w,0,'cpu_fallback')
   x=torch.randn(1,65,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(256,device='cuda',dtype=torch.bfloat16)[::2];primitive(x,w,0,'strided_weight_fallback');w=torch.randn(128,device='cuda',dtype=torch.float64);primitive(x,w,0,'other_weight_dtype_fallback');w=torch.randn(1,128,device='cuda',dtype=torch.float32);primitive(x,w,0,'broadcast_weight_fallback');x=torch.randn(17,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);primitive(x,w,0,'rank2_fallback')
   del x,w
  with torch.no_grad():
   x=torch.randn(1,65,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);primitive(x,w,0,'no_grad_fallback');del x,w
  with torch.enable_grad():
   for dtype in [torch.float32,torch.bfloat16]:
    x=torch.randn(1,17,128,device='cuda',dtype=dtype,requires_grad=True);w=torch.randn(128,device='cuda',dtype=dtype,requires_grad=True);x2=x.detach().clone().requires_grad_();w2=w.detach().clone().requires_grad_();a=original(x,w);calls[0]=0;b=candidate(x2,w2);assert calls[0]==0;bits(a,b);a.float().square().sum().backward();b.float().square().sum().backward();bits(x.grad,x2.grad);bits(w.grad,w2.grad);d['gradient_cases']+=1;del x,w,x2,w2,a,b
  with torch.inference_mode():
   x=torch.randn(1,65,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16);expected=original(x,w);previous_import=builtins.__import__
   for missing,should_fallback in [('triton',True),('triton.language',True),('unrelated_dependency',False)]:
    def blocked(name,*args,**kwargs):
     if name=='spark_engine.qwen_norm_tail':raise ModuleNotFoundError('simulated dependency error mentioning triton',name=missing)
     return previous_import(name,*args,**kwargs)
    builtins.__import__=blocked;calls[0]=0
    try:
     if should_fallback:bits(candidate(x,w),expected)
     else:
      try:candidate(x,w)
      except ModuleNotFoundError as e:assert e.name==missing
      else:raise AssertionError('Unrelated missing module swallowed')
     assert calls[0]==0
    finally:builtins.__import__=previous_import
   d['missing_triton_only_falls_back']=True
   for error in [RuntimeError('norm-tail-sentinel'),ModuleNotFoundError('helper runtime import failed',name='triton.runtime')]:
    def failed(*args):raise error
    tail.norm_tail=failed
    try:candidate(x,w)
    except BaseException as e:assert e is error
    else:raise AssertionError('Helper execution error swallowed')
    finally:tail.norm_tail=observed
    error.__traceback__=None
   del error,failed
   d['helper_errors_propagate']=True;del x,w,expected
   with tempfile.TemporaryDirectory() as tmp:
    select(candidate);fixture.create_fixture(Path(tmp));gc.collect();torch.cuda.empty_cache();model=fixture.load_quantized_qwen(tmp)
    for batch in [1,2]:
     for length in [1,15,16,17,63,64,65,255,256,257]:model_case(model,batch,length)
    del model
 finally:select(candidate);tail.norm_tail=actual_tail
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['primitive_cases'])==86 and len(d['model_cases'])==20 and d['gradient_cases']==2;d['status']='complete';save();print('NORM_INTEGRATION_EXACT_PASSED primitives='+str(len(d['primitive_cases']))+' models='+str(len(d['model_cases'])),flush=True)
if __name__=='__main__':main()
