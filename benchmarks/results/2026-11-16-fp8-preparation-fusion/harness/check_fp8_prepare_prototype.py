"""Strict reference scale/byte comparison and isolated timing; no model/GEMM implementation."""
from pathlib import Path
import sys,json,importlib.util,gc,statistics,os,hashlib
sys.path.insert(0,'/workspace')
import torch
run,mode,main=sys.argv[1:4];out=Path('/results')/run;out.mkdir(exist_ok=True);root=Path('/workspace/benchmark_runs')/main;os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache');torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
spec=importlib.util.spec_from_file_location('fp8_prep_proto',root/'prep_proto.py');candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
d={'status':'running','mode':mode,'scope':'Synthetic FP8 preparation exact scales/bytes and isolated timings only; no real model or generation','candidate_sha256':hashlib.sha256((root/'prep_proto.py').read_bytes()).hexdigest(),'cases':[],'timings':[]}
import triton.language as tl
d['installed_triton_api']={name:hasattr(tl,name) for name in ['any','isnan','sum','max','div_rn']}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
def reference(x):
 s=torch.clamp(x.abs().amax(-1,keepdim=True).float()/448.0,min=1e-6);q=(x/s).to(torch.float8_e4m3fn);q=q.contiguous();pad=(-x.shape[0])%16
 if pad:q=torch.nn.functional.pad(q,(0,0,0,pad));s=torch.nn.functional.pad(s,(0,0,0,pad))
 return q,s

def check(x,label):
 before=x.clone();q,s=reference(x);actual=candidate.prepare_fp8(x);assert actual is not None,label+' unexpectedly unsupported';aq,as_=actual;assert aq.shape==q.shape and as_.shape==s.shape and aq.dtype==q.dtype and as_.dtype==s.dtype and aq.is_contiguous() and as_.is_contiguous();sd=(s.view(torch.int32)!=as_.view(torch.int32));qd=(q.view(torch.uint8)!=aq.view(torch.uint8));row={'label':label,'shape':list(x.shape),'scale_differences':int(sd.sum()),'byte_differences':int(qd.sum())};d['cases'].append(row);save()
 if bool(sd.any()) or bool(qd.any()):
  si=sd.nonzero()[:8];qi=qd.nonzero()[:8];d['failure']={'classification':'scale' if bool(sd.any()) else 'fp8_bytes','case':label,'scale_examples':[{'index':i,'expected_bits':hex(int(s.view(torch.int32)[tuple(i)].item()) & 0xffffffff),'actual_bits':hex(int(as_.view(torch.int32)[tuple(i)].item()) & 0xffffffff)} for i in si.tolist()],'byte_examples':[{'index':i,'expected':int(q.view(torch.uint8)[tuple(i)].item()),'actual':int(aq.view(torch.uint8)[tuple(i)].item())} for i in qi.tolist()]};torch.save({'input':x.cpu(),'expected_scale':s.cpu(),'actual_scale':as_.cpu(),'expected_bytes':q.view(torch.uint8).cpu(),'actual_bytes':aq.view(torch.uint8).cpu()},out/'first-mismatch.pt');save();raise AssertionError(label+' exact intermediate mismatch')
 assert torch.equal(x.view(torch.int16),before.view(torch.int16));assert aq.data_ptr()!=x.data_ptr() and as_.data_ptr()!=x.data_ptr();del before,q,s,aq,as_,actual,sd,qd

def ms(fn):
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();v=fn();b.record();b.synchronize();t=a.elapsed_time(b);del v;return t

def timing(x):
 for _ in range(5):a=reference(x);b=candidate.prepare_fp8(x);del a,b
 samples=[]
 for backend in ['original','candidate','candidate','original']*8:samples.append({'backend':backend,'ms':ms(lambda:reference(x)) if backend=='original' else ms(lambda:candidate.prepare_fp8(x))})
 med={name:statistics.median(v['ms'] for v in samples if v['backend']==name) for name in ['original','candidate']};return {'shape':list(x.shape),'median_ms':med,'saved_ms':med['original']-med['candidate'],'samples':samples}
try:
 with torch.inference_mode():
  # Exhaustive finite nonnegative BF16 row-max domain in bounded128-row batches.
  values=torch.arange(32640,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
  for start in range(0,32640,128):
   v=values[start:start+128];x=v[:,None].expand(-1,5120).contiguous();check(x,'all_finite_maxima_'+str(start));del x,v
  d['finite_bf16_maxima_checked']=32640;save();del values;print('ALL32640_FINITE_MAXIMA_EXACT',flush=True)
  for width in [5120,6144]:
   for rows in [16,17,31,32,256,1024]:
    for factor in [0.,1e-8,1e-4,1.,32.,1e30]:
     x=(torch.randn(rows,width,device='cuda')*factor).to(torch.bfloat16);check(x,f'random_{rows}_{width}_{factor}');del x
    x=torch.full((rows,width),-0.,device='cuda',dtype=torch.bfloat16);check(x,f'negativezero_{rows}_{width}');del x
   # Boundary coverage: all BF16 mantissas, both signs and relative powers, anchored row maxima.
   mant=1+torch.arange(128,device='cuda',dtype=torch.float32)/128.;levels=torch.cat([mant*(2.**e) for e in range(0,-19,-1)]);levels=torch.cat([levels,-levels]);levels=torch.nn.functional.pad(levels,(0,width-levels.numel()))
   for exponent in [-12,-10,-1,0,10,60,127]:
    maxima=(mant*(2.**exponent)).to(torch.bfloat16);base=(levels*(2.**exponent)).to(torch.bfloat16);x=base[None,:].expand(128,-1).clone();x=torch.maximum(torch.minimum(x,maxima[:,None]),-maxima[:,None]);x[:,0]=maxima;check(x,f'boundaries_{width}_{exponent}');del maxima,base,x
   del mant,levels
   for value in [float('inf'),-float('inf'),float('nan')]:
    x=torch.ones(17,width,device='cuda',dtype=torch.bfloat16);x[:,0]=value;check(x,f'nonfinite_{width}_{value}');del x
   for rows in [256,1024]:
    x=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16);d['timings'].append(timing(x));save();del x
  for label,x in [('wide',torch.zeros(16,17408,device='cuda',dtype=torch.bfloat16)),('short',torch.zeros(1,5120,device='cuda',dtype=torch.bfloat16)),('fp32',torch.zeros(16,5120,device='cuda')),('cpu',torch.zeros(16,5120,dtype=torch.bfloat16)),('strided',torch.zeros(16,10240,device='cuda',dtype=torch.bfloat16)[:,::2])]:assert candidate.prepare_fp8(x) is None,label
  del x
 with torch.no_grad():
  x=torch.zeros(16,5120,device='cuda',dtype=torch.bfloat16);assert candidate.prepare_fp8(x) is None;del x
 d['metadata_fallback_cases']=6
except BaseException as error:
 d['status']='failed';d['error']={'type':type(error).__name__,'message':str(error)};save();raise
spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);gc.collect();d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2;d['status']='complete';save();print('FP8_PREP_PROTOTYPE_EXACT_COMPLETE',flush=True)
