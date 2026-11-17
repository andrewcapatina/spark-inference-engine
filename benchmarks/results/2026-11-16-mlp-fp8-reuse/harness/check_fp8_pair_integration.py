"""Real FP8-layer exactness and module dispatch/fallback checks, not whole-model validation."""
from pathlib import Path
import sys,json,gc,importlib.util,types,statistics,time
sys.path.insert(0,'/workspace')
import torch
from safetensors import safe_open
from spark_engine.qwen_fp8 import FP8Linear
import spark_engine.qwen_mlp_fp8_pair as pairs
import spark_engine.qwen_model as models
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
ns=dict(vars(models));exec(compile(Path('/harness/mlp_forward_before_fp8_pair.py.txt').read_text(),'accepted-activation-reference','exec'),ns);original=ns['forward'];d={'status':'running','mode':mode,'scope':'All8real-weight FP8 MLP layers and dispatch/fallbacks only; not whole-model timing or generation validation','real_layer_cases':[],'fallback_cases':[]};root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];calls=[];real_factory=pairs._make_pair_projector

def observed(*args):
 project=real_factory(*args)
 if project is None:return None
 def call(x):calls.append(list(x.shape));return project(x)
 return call
pairs._make_pair_projector=observed

def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def get(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
def linear(prefix):return FP8Linear(get(prefix+'.weight'),get(prefix+'.weight_scale'))
def exact(a,b,label):
 if not torch.equal(a.view(torch.int16),b.view(torch.int16)):
  torch.save({'expected':a.cpu(),'actual':b.cpu()},out/(label+'-mismatch.pt'));raise AssertionError(label+' bitwise mismatch; tensors retained')
with torch.inference_mode():
 for layer_index in range(56,64):
  with torch.device('meta'):layer=models.QwenMLP({'hidden_size':5120,'intermediate_size':17408})
  prefix=f'model.language_model.layers.{layer_index}.mlp.';layer.gate_proj=linear(prefix+'gate_proj');layer.up_proj=linear(prefix+'up_proj');layer.down_proj=linear(prefix+'down_proj');project=real_factory(layer.gate_proj,layer.up_proj);assert project is not None
  for rows in [1,15,16,17,256,1024]:
   x=torch.randn(1,rows,5120,device='cuda',dtype=torch.bfloat16);before=x.clone();a,b=layer.gate_proj(x),layer.up_proj(x);pa,pb=project(x);exact(a,pa,'gate');exact(b,pb,'up');ref=original(layer,x);calls.clear();actual=layer(x);exact(ref,actual,'layer');assert len(calls)==int(rows>=16);assert torch.equal(x,before);assert pa.data_ptr()!=pb.data_ptr();d['real_layer_cases'].append({'layer':layer_index,'rows':rows,'pair_exact':True,'block_exact':True,'reuse_calls':len(calls)});del x,before,a,b,pa,pb,ref,actual
  del project,layer;gc.collect();torch.cuda.empty_cache();save();print('REAL_FP8_LAYER_EXACT '+str(layer_index),flush=True)
 # Tiny real FP8 modules allow focused hook/override/subclass checks without extra large tensors.
 def small(n,k):return FP8Linear(torch.randn(n,k,device='cuda').to(torch.float8_e4m3fn),torch.ones(n,1,device='cuda',dtype=torch.bfloat16))
 with torch.device('meta'):layer=models.QwenMLP({'hidden_size':16,'intermediate_size':32})
 layer.gate_proj=small(32,16);layer.up_proj=small(32,16);layer.down_proj=small(16,32)
 for label in ['normal','gate_forward_hook','up_pre_hook','instance_forward','noncontiguous','batch2','decode']:
  x=torch.randn(1,16,16,device='cuda',dtype=torch.bfloat16);handle=None
  if label=='gate_forward_hook':handle=layer.gate_proj.register_forward_hook(lambda m,args,out:out+.125)
  if label=='up_pre_hook':handle=layer.up_proj.register_forward_pre_hook(lambda m,args:(args[0]*.5,))
  if label=='instance_forward':
   old_forward=layer.gate_proj.forward
   def custom(self,x):return old_forward(x)+.125
   layer.gate_proj.forward=types.MethodType(custom,layer.gate_proj)
  if label=='noncontiguous':x=torch.randn(1,16,32,device='cuda',dtype=torch.bfloat16)[:,:,::2]
  if label=='batch2':x=x.expand(2,-1,-1).contiguous()
  if label=='decode':x=x[:,:1].contiguous()
  ref=original(layer,x);calls.clear();actual=layer(x);exact(ref,actual,label);assert len(calls)==int(label=='normal');d['fallback_cases'].append({'case':label,'exact':True,'reuse_calls':len(calls)})
  if handle is not None:handle.remove()
  if label=='instance_forward':del layer.gate_proj.forward;del old_forward
  del x,ref,actual
 class DerivedFP8(FP8Linear):pass
 derived=DerivedFP8(layer.gate_proj.weight,layer.gate_proj.weight_scale);assert real_factory(derived,layer.up_proj) is None;assert real_factory(layer.gate_proj,small(16,16)) is None;d['subclass_and_shape_rejections']=2;del derived,layer
# Standard dense modules and autograd still follow the accepted path.
for device,dtype in [('cpu',torch.float32),('cpu',torch.bfloat16),('cuda',torch.bfloat16)]:
 layer=models.QwenMLP({'hidden_size':16,'intermediate_size':32}).to(device=device,dtype=dtype);x=torch.randn(1,16,16,device=device,dtype=dtype,requires_grad=True);calls.clear();a=original(layer,x);b=layer(x);assert torch.equal(a,b) and not calls;inputs=(x,*layer.parameters());ag=torch.autograd.grad(a.sum(),inputs);bg=torch.autograd.grad(b.sum(),inputs);assert all(torch.equal(i,j) for i,j in zip(ag,bg));del layer,x,a,b,inputs,ag,bg
pairs._make_pair_projector=real_factory;d['gradient_cases']=3;gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('FP8_PAIR_INTEGRATION_COMPLETE',flush=True)
