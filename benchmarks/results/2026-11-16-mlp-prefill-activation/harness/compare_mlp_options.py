"""Independent reference checks and measurements; implementation comes from local Qwen."""
from pathlib import Path
import sys,json,gc,statistics,importlib.util,os,hashlib,time
sys.path.insert(0,'/workspace')
import torch
from safetensors import safe_open
from spark_engine.qwen_nvfp4 import NVFP4Linear
from spark_engine.qwen_fp8 import FP8Linear
run,mode,main=sys.argv[1:4];out=Path('/results')/run;out.mkdir(exist_ok=True);src=Path('/workspace/benchmark_runs')/main;os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache');torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,src/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
nv=module('nv_pair','nv_pair.py');fp=module('fp_pair','fp_pair.py');act=module('mlp_act','activation.py')
d={'status':'running','mode':mode,'scope':'Isolated real-weight MLP layer prototypes and synthetic exhaustive BF16 activation; no whole-model speed/quality claim','candidate_sha256':{f:hashlib.sha256((src/f).read_bytes()).hexdigest() for f in ['nv_pair.py','fp_pair.py','activation.py']},'activation_cases':[],'layer_cases':[]}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def exact(a,b,label):
 finite=torch.isfinite(a)&torch.isfinite(b);same_mask=torch.equal(torch.isnan(a),torch.isnan(b)) and torch.equal(torch.isinf(a),torch.isinf(b));bits=(a.view(torch.int16)==b.view(torch.int16)) | (torch.isnan(a)&torch.isnan(b))
 if not same_mask or not bool(bits.all()):
  torch.save({'expected':a.cpu(),'actual':b.cpu()},out/(label+'-mismatch.pt'));raise AssertionError(label+' element/rounding mismatch; retained tensors')
def measure(fn):
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();x=fn();b.record();b.synchronize();ms=a.elapsed_time(b);del x;return ms
def abba(original,candidate):
 for _ in range(5):a=original();b=candidate();del a,b
 samples=[]
 for name in ['original','candidate','candidate','original']*8:samples.append({'backend':name,'ms':measure(original if name=='original' else candidate)})
 med={name:statistics.median(x['ms'] for x in samples if x['backend']==name) for name in ['original','candidate']};return {'median_ms':med,'saved_ms':med['original']-med['candidate'],'less_time_pct':100*(1-med['candidate']/med['original']),'samples':samples}
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def get(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).to('cuda')
def linear(prefix,backend):
 if backend=='nvfp4':return NVFP4Linear(get(prefix+'.weight_packed'),get(prefix+'.weight_scale'),get(prefix+'.weight_global_scale'),get(prefix+'.input_global_scale'))
 return FP8Linear(get(prefix+'.weight'),get(prefix+'.weight_scale'))
with torch.inference_mode():
 try:
  # All 65536 BF16 bit patterns: verifies the unary rounding map, including special values.
  g=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
  for multiplier in [1.,-1.,0.,-0.,.5,3.25,3.3895313892515355e38,1.1754943508222875e-38,9.183549615799121e-41]:
   u=torch.full_like(g,multiplier);a=torch.nn.functional.silu(g)*u;b=act.silu_mul(g,u);exact(a,b,'exhaustive-bf16-'+str(multiplier));d['activation_cases'].append({'case':'all65536BF16','up':str(multiplier),'exact':True});del u,a,b
  del g
  for shape in [(1,),(17,),(2,3,129),(1,256,17408),(1,1024,17408)]:
   g=torch.randn(shape,device='cuda',dtype=torch.bfloat16);u=torch.randn_like(g);before_g=g.clone();before_u=u.clone();a=torch.nn.functional.silu(g)*u;b=act.silu_mul(g,u);exact(a,b,'activation-'+str(len(d['activation_cases'])));assert torch.equal(g,before_g) and torch.equal(u,before_u) and b.data_ptr() not in [g.data_ptr(),u.data_ptr()];row={'shape':list(shape),'exact':True}
   if len(shape)==3 and shape[-1]==17408:row['timing']=abba(lambda:torch.nn.functional.silu(g)*u,lambda:act.silu_mul(g,u))
   d['activation_cases'].append(row);save();del g,u,before_g,before_u,a,b
 except AssertionError as error:
  d['activation_gate']={'status':'rejected','error':str(error)};save();print('ACTIVATION_NUMERICAL_REJECTED '+str(error),flush=True)
 else:d['activation_gate']={'status':'passed'}
 for backend,layer,factory in [('nvfp4',0,nv._make_pair_projector),('fp8',56,fp._make_pair_projector)]:
  prefix=f'model.language_model.layers.{layer}.mlp.';gate=linear(prefix+'gate_proj',backend);up=linear(prefix+'up_proj',backend);down=linear(prefix+'down_proj',backend);torch.cuda.synchronize();start=time.perf_counter();project=factory(gate,up);torch.cuda.synchronize();setup_ms=(time.perf_counter()-start)*1000;assert project is not None
  if backend=='nvfp4':
   bad_up=NVFP4Linear(up.weight_packed,up.weight_scale,up.weight_global_scale,up.input_global_scale*2);assert factory(gate,bad_up) is None;del bad_up
  assert factory(torch.nn.Linear(4,4),torch.nn.Linear(4,4)) is None
  for rows in [1,15,16,17,256,1024]:
   x=torch.randn((1,rows,5120),device='cuda',dtype=torch.bfloat16);g,u=gate(x),up(x);cg,cu=project(x);exact(g,cg,backend+'-gate-'+str(rows));exact(u,cu,backend+'-up-'+str(rows));expected=down(torch.nn.functional.silu(g)*u);actual=down(torch.nn.functional.silu(cg)*cu);exact(expected,actual,backend+'-reuse-block-'+str(rows));row={'backend':backend,'layer':layer,'rows':rows,'pair_element_exact':True,'pair_reuse_block_element_exact':True,'factory_setup_wall_ms_excluded_from_timing':setup_ms}
   if d['activation_gate']['status']=='passed':
    fused=down(act.silu_mul(cg,cu));exact(expected,fused,backend+'-fused-block-'+str(rows));row['combined_block_element_exact']=True;del fused
   if rows in [256,1024]:
    for repetition in range(8):
     rg,ru=project(x);exact(g,rg,backend+'-repeat-gate-'+str(repetition));exact(u,ru,backend+'-repeat-up-'+str(repetition));del rg,ru
    row['repeatability_exact_repetitions']=8
    def baseline():return down(torch.nn.functional.silu(gate(x))*up(x))
    def reuse():
     a,b=project(x);return down(torch.nn.functional.silu(a)*b)
    def activation():return down(act.silu_mul(gate(x),up(x)))
    def both():
     a,b=project(x);return down(act.silu_mul(a,b))
    row['pair_only']=abba(lambda:(gate(x),up(x)),lambda:project(x));row['mlp_reuse_only']=abba(baseline,reuse);
    if d['activation_gate']['status']=='passed':row['mlp_activation_only']=abba(baseline,activation);row['mlp_combined']=abba(baseline,both)
   d['layer_cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ['pair_only','mlp_reuse_only','mlp_activation_only','mlp_combined']}),flush=True);del x,g,u,cg,cu,expected,actual
  if rows in [256,1024]:del baseline,reuse,activation,both
  del gate,up,down,project;gc.collect();torch.cuda.empty_cache()
d['pair_reuse_gate']={'status':'passed'}
# GPU fallback paths preserve the original expression.
for use_no_grad,dtype,strided in [(True,torch.bfloat16,False),(False,torch.float32,False),(False,torch.bfloat16,True)]:
 with torch.no_grad() if use_no_grad else torch.inference_mode():
  g=torch.randn(3,8,device='cuda',dtype=dtype);u=torch.randn_like(g)
  if strided:g=g[:,::2];u=u[:,::2]
  assert torch.equal(act.silu_mul(g,u),torch.nn.functional.silu(g)*u)
 del g,u
d['gpu_fallback_cases_exact']=3
# CPU and autograd paths must retain original expression, never bypass gradients.
for dtype in [torch.float32,torch.bfloat16]:
 g=torch.randn(3,7,dtype=dtype,requires_grad=True);u=torch.randn_like(g,requires_grad=True);a=act.silu_mul(g,u);b=torch.nn.functional.silu(g)*u;assert torch.equal(a,b);ag=torch.autograd.grad(a.sum(),(g,u));bg=torch.autograd.grad(b.sum(),(g,u));assert all(torch.equal(x,y) for x,y in zip(ag,bg))
d['cpu_gradient_fallback_exact']=True;del g,u,a,b,ag,bg;gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('MLP_OPTIONS_COMPONENT_COMPARISON_COMPLETE',flush=True)
