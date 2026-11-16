"""Isolated local-model helper against retained eager gated RMSNorm; no model integration."""
from pathlib import Path
import os,sys,json,time,gc,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 import torch.nn.functional as F
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache')
 from spark_engine.qwen_gated_norm_tail import gated_norm_tail
 ns={};exec(compile(Path('/harness/qwen_ops_before_gated_processing.py.txt').read_text(),'retained-original-gated-ops','exec'),ns);original=ns['gated_rms_norm'];torch.set_num_threads(4);torch.manual_seed(36036);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'running','mode':mode,'scope':'Standalone gated pointwise tail only; original eager FP32 statistics and SiLU captured observationally. Not real-checkpoint or integration validation.','cases':[],'first_use_records':[],'triton_cache_dir':os.environ['TRITON_CACHE_DIR'],'invalid_cases':[]};seen=set()
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def bits(a,b):
  assert a.shape==b.shape and a.dtype==b.dtype
  dt=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
  assert torch.equal(a.contiguous().view(dt),b.contiguous().view(dt)),'Bitwise result changed'
 def case(shape,variant='random',given_x=None,given_z=None,parameter=False):
  x=torch.randn(shape,device='cuda',dtype=torch.bfloat16) if given_x is None else given_x
  z=torch.randn(shape,device='cuda',dtype=torch.bfloat16) if given_z is None else given_z
  w=torch.randn(shape[-1],device='cuda',dtype=torch.bfloat16)*.1
  if variant=='zeros':x.zero_()
  elif variant=='zero_weight':w.zero_()
  elif variant=='unit_weight':w.fill_(1)
  elif variant=='negative_weight':w.fill_(-.125)
  elif variant=='bounded_weight':w.fill_(.125)
  if parameter:w=torch.nn.Parameter(w,requires_grad=True)
  snapshots=[t.clone() for t in [x,z,w]];inv=[];gate=[];rsqrt=torch.rsqrt;silu=F.silu
  def observe_inv(a):v=rsqrt(a);inv.append(v);return v
  def observe_gate(a,*args,**kwargs):v=silu(a,*args,**kwargs);gate.append(v);return v
  torch.rsqrt=observe_inv;F.silu=observe_gate
  try:expected=original(x,w,z)
  finally:torch.rsqrt=rsqrt;F.silu=silu
  assert len(inv)==len(gate)==1;args=[x.float().contiguous(),inv[0].contiguous(),w,gate[0].contiguous()];before=[t.clone() for t in args];torch.cuda.synchronize();start=time.perf_counter();actual=gated_norm_tail(*args);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  if shape[-1] not in seen:d['first_use_records'].append({'width':shape[-1],'first_use_seconds':elapsed,'shape':list(shape)});seen.add(shape[-1])
  assert actual.dtype==torch.bfloat16 and actual.is_contiguous() and actual.data_ptr()!=args[0].data_ptr();bits(actual,expected)
  for a,b in zip(args,before):bits(a,b)
  for a,b in zip([x,z,w],snapshots):bits(a,b)
  assert torch.isfinite(actual).all();d['cases'].append({'shape':list(shape),'variant':variant,'x_source_contiguous':x.is_contiguous(),'z_source_contiguous':z.is_contiguous(),'parameter_requires_grad':parameter,'output_and_inputs_bitwise_exact':True});save()
 save()
 with torch.inference_mode():
  for width in [1,16,127,128,129,256]:
   for rows in [1,3,17,63,64,65,256,1024]:
    for variant in ['random','zeros','unit_weight','negative_weight']:case((rows,width),variant)
  for rows in [12288,49152]:case((rows,128))
  for shape in [(1,17,128),(2,65,128),(1,64,48,128)]:
   given=torch.randn(*shape[:-1],shape[-1]*2,device='cuda',dtype=torch.bfloat16)[...,:shape[-1]];case(shape,given_x=given,given_z=given);del given
  patterns=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16);values=patterns[torch.isfinite(patterns)];assert values.numel()==65280;given=values.reshape(-1,128).cuda()
  for variant in ['random','zero_weight','negative_weight']:case(tuple(given.shape),variant,given_x=given)
  case(tuple(given.shape),'bounded_weight',given_z=given);d['finite_bf16_x_patterns_covered']=d['finite_bf16_z_patterns_covered']=values.numel();del patterns,values,given
  signed=torch.tensor([0.,-0.,1e-40,-1e-40,1e-20,-1e-20,1e20,-1e20,1e38,-1e38],device='cuda',dtype=torch.bfloat16).view(10,1).expand(10,128)
  for variant in ['bounded_weight','negative_weight','zero_weight']:case(tuple(signed.shape),variant,given_x=signed,given_z=signed)
  del signed;case((17,128),parameter=True)
  args=[torch.ones(2,128,device='cuda'),torch.ones(2,1,device='cuda'),torch.ones(128,device='cuda',dtype=torch.bfloat16),torch.ones(2,128,device='cuda')]
  invalid=[('weight_length',2,torch.ones(127,device='cuda',dtype=torch.bfloat16)),('weight_fp32',2,args[2].float()),('weight_stride',2,torch.ones(256,device='cuda',dtype=torch.bfloat16)[::2]),('inverse_rows',1,torch.ones(3,1,device='cuda')),('silu_shape',3,torch.ones(1,256,device='cuda')),('x_dtype',0,args[0].bfloat16()),('silu_dtype',3,args[3].bfloat16()),('weight_cpu',2,args[2].cpu())]
  for label,index,value in invalid:
   trial=list(args);trial[index]=value
   try:gated_norm_tail(*trial)
   except AssertionError:d['invalid_cases'].append(label)
   else:raise AssertionError('Invalid helper metadata accepted: '+label)
  del invalid,trial,value
  empty=gated_norm_tail(torch.empty(0,128,device='cuda'),torch.empty(0,1,device='cuda'),args[2],torch.empty(0,128,device='cuda'));assert empty.shape==(0,128) and empty.dtype==torch.bfloat16;del empty
 try:gated_norm_tail(*args)
 except AssertionError:d['outside_inference_mode_rejected']=True
 else:raise AssertionError('Helper accepted non-inference use')
 del args;gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['cases'])==205;d['status']='complete';save();print('GATED_NORM_TAIL_BITWISE_EXACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
