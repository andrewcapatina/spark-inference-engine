"""Standalone local-model tail versus original RMSNorm with captured eager inverse RMS."""
from pathlib import Path
import os,sys,json,time,gc,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache')
 from spark_engine.qwen_norm_tail import norm_tail
 ns={};exec(compile(Path('/harness/qwen_ops_before_norm_processing.py.txt').read_text(),'retained-original-qwen-ops','exec'),ns);original=ns['qwen_rms_norm'];torch.set_num_threads(4);torch.manual_seed(32032);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'running','mode':mode,'scope':'Standalone pointwise-tail only. Original eager RMSNorm supplies inverse RMS through an observational hook; no alternate reduction. Not engine integration or real-checkpoint validation.','cases':[],'first_use_records':[],'triton_cache_dir':os.environ['TRITON_CACHE_DIR']};seen=set()
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def bits(a,b):
  assert a.shape==b.shape and a.dtype==b.dtype
  dtype=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
  assert torch.equal(a.contiguous().view(dtype),b.contiguous().view(dtype)),'Bitwise result changed'
 def case(shape,wdtype,variant='random',given=None,parameter=False):
  x=torch.randn(shape,device='cuda',dtype=torch.bfloat16) if given is None else given;weight=torch.randn(shape[-1],device='cuda',dtype=wdtype)*.1
  if variant=='zeros':x.zero_()
  elif variant=='neg_one_weight':weight.fill_(-1)
  elif variant=='zero_weight':weight.zero_()
  if parameter:weight=torch.nn.Parameter(weight,requires_grad=True)
  x_before=x.clone();w_before=weight.clone();captured=[];previous=torch.rsqrt
  def observe(a):
   value=previous(a);captured.append(value);return value
  torch.rsqrt=observe
  try:expected=original(x,weight)
  finally:torch.rsqrt=previous
  assert len(captured)==1;inverse=captured[0].contiguous();x32=x.float().contiguous();before=x32.clone();inv_before=inverse.clone();key=(shape[-1],str(wdtype));torch.cuda.synchronize();start=time.perf_counter();actual=norm_tail(x32,inverse,weight);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  if key not in seen:d['first_use_records'].append({'width':shape[-1],'weight_dtype':str(wdtype),'first_use_seconds':elapsed,'shape':list(shape)});seen.add(key)
  assert actual.dtype==torch.bfloat16 and actual.is_contiguous();bits(actual,expected);bits(x32,before);bits(inverse,inv_before);bits(x,x_before);bits(weight,w_before);assert torch.isfinite(actual).all();d['cases'].append({'shape':list(shape),'weight_dtype':str(wdtype),'variant':variant,'parameter_requires_grad':parameter,'source_contiguous':x.is_contiguous(),'expected_inverse_shape':list(inverse.shape),'output_and_inputs_bitwise_exact':True});save()
 save()
 with torch.inference_mode():
  for width in [16,128,256,5120]:
   for wdtype in [torch.bfloat16,torch.float32]:
    for batch,length in [(1,1),(1,3),(2,17),(1,63),(1,64),(1,65),(1,256),(1,1024)]:
     for variant in ['random','zeros','neg_one_weight']:case((batch,length,width),wdtype,variant)
  for shape in [(1,64,24,256),(1,256,24,256),(1,1024,24,256),(1,1024,4,256)]:
   for wdtype in [torch.bfloat16,torch.float32]:
    given=torch.randn(*shape[:-1],shape[-1]*2,device='cuda',dtype=torch.bfloat16)[...,:shape[-1]];case(shape,wdtype,given=given)
  del given
  for shape in [(1,128),(17,128),(65,256)]:
   for wdtype in [torch.bfloat16,torch.float32]:case(shape,wdtype)
  patterns=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16);values=patterns[torch.isfinite(patterns)];assert values.numel()==65280;given=values.reshape(1,-1,128).cuda()
  for wdtype in [torch.bfloat16,torch.float32]:
   for variant in ['random','zero_weight','neg_one_weight']:case(tuple(given.shape),wdtype,variant,given=given)
  d['finite_bf16_patterns_covered']=values.numel();del patterns,values,given
  given=torch.tensor([0.,-0.,1e-40,-1e-40,1e-20,-1e-20,1e20,-1e20,1e38,-1e38],device='cuda',dtype=torch.bfloat16).view(1,10,1).expand(1,10,16)
  for wdtype in [torch.bfloat16,torch.float32]:case(tuple(given.shape),wdtype,given=given)
  del given;case((1,17,5120),torch.bfloat16,parameter=True)
  invalid=(torch.ones(2,16,device='cuda',dtype=torch.float32),torch.ones(2,1,device='cuda'),torch.ones(16,device='cuda',dtype=torch.bfloat16))
 try:norm_tail(*invalid)
 except AssertionError:d['outside_inference_mode_rejected']=True
 else:raise AssertionError('Helper silently accepted non-inference use')
 del invalid;gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;assert len(d['cases'])==215;d['status']='complete';save();print('NORM_TAIL_BITWISE_EXACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
