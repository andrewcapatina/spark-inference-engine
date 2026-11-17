"""Strict real-weight integration comparisons against retained accepted implementations."""
from pathlib import Path
import sys,json,gc,types,textwrap,importlib.util,builtins
sys.path.insert(0,'/workspace')
import torch
from safetensors import safe_open
import spark_engine.qwen_fp8 as fp8
import spark_engine.qwen_mlp_fp8_pair as pairs
import spark_engine.qwen_fp8_prepare as prep
import spark_engine.qwen_model as models
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True)
import os
os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
ns=dict(vars(fp8));exec(compile(textwrap.dedent(Path('/harness/fp8_forward_before_prepare.py.txt').read_text()),'retained-forward','exec'),ns);old_forward=ns['forward'];new_forward=fp8.FP8Linear.forward
ns=dict(vars(pairs));exec(compile(Path('/harness/fp8_pair_before_prepare.py.txt').read_text(),'retained-pair','exec'),ns);old_pair=ns['_make_pair_projector'];new_pair=pairs._make_pair_projector
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];d={'status':'running','mode':mode,'scope':'Real-weight integration and dispatch only; not whole-model validation','layer_cases':[],'linear_cases':[],'fallback_cases':[]};calls=[];real_prepare=prep.prepare_fp8

def observed(x):
 result=real_prepare(x);calls.append({'shape':list(x.shape),'fused':result is not None});return result
prep.prepare_fp8=observed

def select(new):fp8.FP8Linear.forward=new_forward if new else old_forward;pairs._make_pair_projector=new_pair if new else old_pair
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def exact(a,b,label):
 if not torch.equal(a.view(torch.int16),b.view(torch.int16)):
  torch.save({'expected':a.cpu(),'actual':b.cpu()},out/'mismatch.pt');raise AssertionError(label+' output bits differ')
def get(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
def linear(prefix):return fp8.FP8Linear(get(prefix+'.weight'),get(prefix+'.weight_scale'))
try:
 with torch.inference_mode():
  # All eight real FP8 MLPs, including their wide down-projection fallback.
  for layer_index in range(56,64):
   with torch.device('meta'):layer=models.QwenMLP({'hidden_size':5120,'intermediate_size':17408})
   prefix=f'model.language_model.layers.{layer_index}.mlp.';layer.gate_proj=linear(prefix+'gate_proj');layer.up_proj=linear(prefix+'up_proj');layer.down_proj=linear(prefix+'down_proj')
   for rows in [1,15,16,17,256,1024]:
    x=torch.randn(1,rows,5120,device='cuda',dtype=torch.bfloat16);before=x.clone();select(False);reference=layer(x);select(True);calls.clear();actual=layer(x);exact(reference,actual,'MLP');assert torch.equal(x,before);fused=sum(c['fused'] for c in calls);assert fused==int(rows>=16),calls
    d['layer_cases'].append({'layer':layer_index,'rows':rows,'exact':True,'fused_preparations':fused,'preparation_calls':list(calls)});del x,before,reference,actual
   del layer;gc.collect();torch.cuda.empty_cache();save();print('REAL_MLP_EXACT '+str(layer_index),flush=True)
  # Real 6144-input FP8 projection outside the MLP; locate by safetensors header.
  prefix=None
  for shard in sorted(set(index.values())):
   with safe_open(str(root/shard),framework='pt',device='cpu') as f:
    for name in f.keys():
     if name.endswith('.weight') and f.get_slice(name).get_dtype()=='F8_E4M3' and f.get_slice(name).get_shape()[1:]==[6144]:prefix=name[:-7];break
   if prefix:break
  assert prefix is not None;layer=linear(prefix)
  for rows in [1,15,16,17,256,1024]:
   x=torch.randn(rows,6144,device='cuda',dtype=torch.bfloat16);reference=old_forward(layer,x);calls.clear();actual=new_forward(layer,x);exact(reference,actual,'width6144');assert sum(c['fused'] for c in calls)==int(rows>=16);d['linear_cases'].append({'prefix':prefix,'rows':rows,'exact':True,'preparations':list(calls)});del x,reference,actual
  del layer;gc.collect()
  # Metadata/mode dispatch, missing optional backend, and propagation of actual helper errors.
  layer=fp8.FP8Linear(torch.randn(32,5120,device='cuda').to(torch.float8_e4m3fn),torch.ones(32,1,device='cuda'))
  up=fp8.FP8Linear(layer.weight.clone(),layer.weight_scale.clone());project=new_pair(layer,up);old_project=old_pair(layer,up)
  x=torch.randn(17,5120,device='cuda',dtype=torch.bfloat16)
  for target,reference in [(lambda:new_forward(layer,x),lambda:old_forward(layer,x)),(lambda:project(x),lambda:old_project(x))]:
   expected=reference();saved_import=builtins.__import__
   for absent in ["triton","triton.runtime.driver"]:
    def missing(name,*a,**kw):
     if name=='spark_engine.qwen_fp8_prepare':raise ModuleNotFoundError('optional backend absent',name=absent)
     return saved_import(name,*a,**kw)
    try:builtins.__import__=missing;actual=target()
    finally:builtins.__import__=saved_import
    if isinstance(actual,tuple):
     for a,b in zip(expected,actual):exact(a,b,'missing-triton')
    else:exact(expected,actual,'missing-triton')
    d['fallback_cases'].append({'case':'missing-'+absent,'exact':True})
   def fail(x):raise RuntimeError('sentinel helper execution error')
   prep.prepare_fp8=fail
   try:
    try:target()
    except RuntimeError as error:assert str(error)=='sentinel helper execution error'
    else:raise AssertionError('runtime helper error swallowed')
   finally:prep.prepare_fp8=observed
   def wrong(name,*a,**kw):
    if name=='spark_engine.qwen_fp8_prepare':raise ModuleNotFoundError('missing implementation',name='spark_engine.qwen_fp8_prepare')
    return saved_import(name,*a,**kw)
   try:
    builtins.__import__=wrong
    try:target()
    except ModuleNotFoundError as error:assert error.name=='spark_engine.qwen_fp8_prepare'
    else:raise AssertionError('implementation import error swallowed')
   finally:builtins.__import__=saved_import
   del actual,expected
  for label,xx in [('noncontiguous',torch.randn(17,10240,device='cuda',dtype=torch.bfloat16)[:,::2]),('short',x[:1])]:
   calls.clear();a=old_forward(layer,xx);b=new_forward(layer,xx);exact(a,b,label);assert not any(c['fused'] for c in calls);d['fallback_cases'].append({'case':label,'exact':True});del a,b,xx
  del target,reference,project,old_project,layer,up,x
 with torch.no_grad():
  layer=fp8.FP8Linear(torch.randn(32,5120,device='cuda').to(torch.float8_e4m3fn),torch.ones(32,1,device='cuda'));x=torch.randn(17,5120,device='cuda',dtype=torch.bfloat16);calls.clear();a=old_forward(layer,x);b=new_forward(layer,x);exact(a,b,'no_grad');assert not calls;d['fallback_cases'].append({'case':'no_grad','exact':True});del layer,x,a,b
finally:select(True);prep.prepare_fp8=real_prepare
spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);gc.collect();d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('FP8_PREPARATION_INTEGRATION_EXACT_COMPLETE',flush=True)
