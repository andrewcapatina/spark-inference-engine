"""Dispatch/fallback integration checks against the retained original MLP method."""
from pathlib import Path
import sys,json,gc,importlib.util,builtins
sys.path.insert(0,'/workspace')
import torch
import spark_engine.qwen_model as models
import spark_engine.qwen_mlp_activation as activation
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
ns=dict(vars(models));exec(compile(Path('/harness/mlp_forward_before_activation.py.txt').read_text(),'retained-mlp-reference','exec'),ns);original=ns['forward'];d={'status':'running','mode':mode,'scope':'Tiny GPU/CPU MLP integration, dispatch, gradients and import errors only; not checkpoint validation','cases':[]};calls=[];real_activation=activation.silu_mul
def observed(g,u):calls.append(list(g.shape));return real_activation(g,u)
activation.silu_mul=observed
with torch.inference_mode():
 for device in ['cpu','cuda']:
  for dtype in [torch.float32,torch.bfloat16]:
   layer=models.QwenMLP({'hidden_size':32,'intermediate_size':64}).to(device=device,dtype=dtype)
   for batch in [1,2]:
    for length in [0,1,15,16,17,64]:
     x=torch.randn(batch,length,32,device=device,dtype=dtype);before=x.clone();calls.clear();a=original(layer,x);b=layer(x);assert torch.equal(a,b) and torch.equal(x,before);eligible=device=='cuda' and dtype==torch.bfloat16 and batch==1 and length>=16;assert len(calls)==int(eligible);d['cases'].append({'device':device,'dtype':str(dtype),'batch':batch,'length':length,'fast_calls':len(calls),'exact':True});del x,before,a,b
   del layer
 # Same import policy as accepted helpers: only missing Triton may fall back.
 layer=models.QwenMLP({'hidden_size':32,'intermediate_size':64}).to('cuda',dtype=torch.bfloat16);x=torch.randn(1,16,32,device='cuda',dtype=torch.bfloat16);expected=original(layer,x);import_impl=builtins.__import__
 for name in ['triton','triton.runtime','unrelated_backend',None]:
  def missing(module,*a,**kw):
   if module=='spark_engine.qwen_mlp_activation':raise ModuleNotFoundError('triton mentioned in unrelated message',name=name)
   return import_impl(module,*a,**kw)
  try:
   builtins.__import__=missing
   try:actual=layer(x)
   except ModuleNotFoundError:assert name in ['unrelated_backend',None]
   else:assert name in ['triton','triton.runtime'] and torch.equal(expected,actual);del actual
  finally:builtins.__import__=import_impl
 def fail(*a):raise RuntimeError('sentinel runtime/compiler failure')
 activation.silu_mul=fail
 try:layer(x)
 except RuntimeError as error:assert str(error)=='sentinel runtime/compiler failure'
 else:raise AssertionError('Unrelated runtime error swallowed')
 activation.silu_mul=observed;del layer,x,expected
for device,dtype in [('cpu',torch.float32),('cuda',torch.bfloat16)]:
 layer=models.QwenMLP({'hidden_size':32,'intermediate_size':64}).to(device=device,dtype=dtype);x=torch.randn(1,16,32,device=device,dtype=dtype,requires_grad=True);calls.clear();a=original(layer,x);b=layer(x);assert torch.equal(a,b) and not calls;inputs=(x,*layer.parameters());ag=torch.autograd.grad(a.sum(),inputs);bg=torch.autograd.grad(b.sum(),inputs);assert all(torch.equal(i,j) for i,j in zip(ag,bg));del layer,x,a,b,inputs,ag,bg
activation.silu_mul=real_activation;d.update(gradient_cases=2,missing_triton_fallback_cases=2,unrelated_import_errors_propagated=2,runtime_errors_propagated=True);gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';(out/'results.json').write_text(json.dumps(d,indent=2)+'\n');print('MLP_INTEGRATION_EXACT cases='+str(len(d['cases'])),flush=True)
