"""Bitwise original-layout key products; primitive timings are not model gains."""
from pathlib import Path
import sys,json,os,statistics,gc
import time as timer
sys.path.insert(0,'/workspace')
import torch
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache')
from spark_engine.qwen_key_prepare import prepare_keys,_make_key_preparer
torch.set_num_threads(4);torch.manual_seed(6060);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
d={'status':'running','mode':mode,'scope':'Standalone original-layout FP32 key products, not checkpoint validation','cases':[],'timings':[],'invalid_cases':[]}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def reference(k,b,e):return k*b.unsqueeze(-1),(k*b.unsqueeze(-1))*e.unsqueeze(-1)
def bits(a,b):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)),'FP32 bits differ'
def check(k,b,e,label):
 before=[x.clone() for x in [k,b,e]];expected=reference(k,b,e);torch.cuda.synchronize();start=timer.perf_counter();actual=prepare_keys(k,b,e);torch.cuda.synchronize()
 if 'first_use_seconds' not in d:d['first_use_seconds']=timer.perf_counter()-start
 for x,y in zip(actual,expected):
  bits(x,y)
  assert all(x.stride(i)==y.stride(i) for i,size in enumerate(x.shape) if size>1),'Physical layout changed'
 if k.numel():assert actual[0].untyped_storage().data_ptr()!=actual[1].untyped_storage().data_ptr() and all(y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in [k,b,e] for y in actual)
 for x,y in zip([k,b,e],before):bits(x,y)
 d['cases'].append({'label':label,'shape':list(k.shape),'input_strides':[list(x.stride()) for x in [k,b,e]],'output_strides':[list(x.stride()) for x in actual],'reference_strides':[list(x.stride()) for x in expected],'bitwise_and_layout_and_ownership_exact':True});save()
with torch.inference_mode():
 for batch,heads in [(1,48),(1,1),(2,3)]:
  for time in [64,1,3,65]:
   for width in [128,1,16,129]:
    for layout in ['time_major','head_major']:
     if layout=='time_major':k=torch.randn(batch,time*2,heads,width,device='cuda').transpose(1,2)[:,:,:time]
     else:k=torch.randn(batch,heads,time*2,width,device='cuda')[:,:,:time]
     beta=torch.rand(batch,time*2,heads*2,device='cuda')[:,::2,::2].transpose(1,2);e=torch.rand(batch,heads,time,device='cuda');check(k,beta,e,layout)
 edge=torch.tensor([0.,-0.,1.401298464324817e-45,-1.401298464324817e-45,1.1754942106924411e-38,-1.1754942106924411e-38,1.1754943508222875e-38,-1.1754943508222875e-38,3.4028234663852886e38,-3.4028234663852886e38],device='cuda').reshape(1,1,1,10)
 for sign in [.5,-.5]:check(edge,torch.full((1,1,1),sign,device='cuda'),torch.full((1,1,1),.5,device='cuda'),'finite_extremes_signedzero_subnormal')
 k=torch.randn(1,64,48,128,device='cuda').transpose(1,2);beta=torch.rand(1,48,64,device='cuda');e=torch.rand_like(beta)
 d['reassociation_counterexample_elements']=int(torch.count_nonzero((k*beta.unsqueeze(-1))*e.unsqueeze(-1)!=k*(beta*e).unsqueeze(-1)));assert d['reassociation_counterexample_elements']>0
 check(k,beta,e,'ordered_rounding');check(k,beta[:1,:1,:1].expand_as(beta),e[:1,:1,:1].expand_as(e),'broadcast_beta_and_decay')
 check(torch.empty(0,64,48,128,device='cuda').transpose(1,2),torch.empty(0,48,64,device='cuda'),torch.empty(0,48,64,device='cuda'),'empty_batch')
 for layout in ['time_major','head_major']:
  for batch,heads,width in [(1,48,128),(2,3,16)]:
   full=torch.randn(batch,256,heads,width,device='cuda').transpose(1,2) if layout=='time_major' else torch.randn(batch,heads,256,width,device='cuda')
   full_beta=torch.rand(batch,256,heads,device='cuda').transpose(1,2);planned=_make_key_preparer(full,full_beta,64);kept=[]
   for start in range(0,256,64):
    kk=full[:,:,start:start+64];bb=full_beta[:,:,start:start+64];ee=torch.rand(batch,heads,64,device='cuda');got=planned(kk,bb,ee);expected=reference(kk,bb,ee)
    for x,y in zip(got,expected):bits(x,y)
    kept.extend(got)
   assert len({x.untyped_storage().data_ptr() for x in kept})==8
   d.setdefault('planned_multichunk_cases',[]).append({'layout':layout,'batch':batch,'heads':heads,'width':width,'chunks':4,'outputs_exact_and_request_owned':True})
   del full,full_beta,planned,kept,got,expected,kk,bb,ee,x,y
 for fn in [reference,prepare_keys]:
  for _ in range(3):result=fn(k,beta,e);del result
 for label in ['original','candidate','candidate','original']*4:
  fn=reference if label=='original' else prepare_keys;a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(32):result=fn(k,beta,e)
  z.record();z.synchronize();d['timings'].append({'backend':label,'cuda_ms_per_prepare':a.elapsed_time(z)/32});del result
 d['median_cuda_ms']={label:statistics.median(x['cuda_ms_per_prepare'] for x in d['timings'] if x['backend']==label) for label in ['original','candidate']}
 args=[k,beta,e]
 for label,index,value in [('cpu',0,k.cpu()),('dtype',1,beta.bfloat16()),('rank',0,k.squeeze(0)),('prefix',2,e[:,:1]),('unsupported_stride',0,torch.randn(1,48,64,256,device='cuda')[...,::2])]:
  trial=list(args);trial[index]=value
  try:prepare_keys(*trial)
  except AssertionError:d['invalid_cases'].append(label)
  else:raise AssertionError('Invalid metadata accepted: '+label)
 huge=torch.ones(1,1,1,128,device='cuda').expand(2**25,1,1,128);huge_beta=torch.ones(1,1,1,device='cuda').expand(2**25,1,1)
 try:prepare_keys(huge,huge_beta,huge_beta)
 except AssertionError:d['invalid_cases'].append('int32_output_limit')
 else:raise AssertionError('Output size guard missing')
 del trial,value,huge,huge_beta
try:prepare_keys(*args)
except AssertionError:d['outside_inference_mode_rejected']=True
else:raise AssertionError('Inference guard missing')
del args,k,beta,e,edge,a,z
gc.collect();torch.cuda.empty_cache();assert torch.cuda.memory_allocated()<=16*1024**2;d['status']='complete';save();print(json.dumps({'cases':len(d['cases']),'median_cuda_ms':d['median_cuda_ms'],'reassociation_counterexamples':d['reassociation_counterexample_elements']}),flush=True)
