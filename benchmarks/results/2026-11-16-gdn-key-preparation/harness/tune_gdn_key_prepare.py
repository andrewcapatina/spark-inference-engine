"""Separate kernel GPU cost from Python checks/launch overhead; no model load."""
from pathlib import Path
import sys,os,json,statistics,gc
sys.path.insert(0,'/workspace')
import torch
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
from spark_engine.qwen_key_prepare import prepare_keys,_key_prepare,_make_key_preparer
torch.set_num_threads(4);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
d={'status':'running','mode':mode,'scope':'Primitive kernel versus wrapper overhead; graph replay only diagnostic, not engine implementation','methods':{}}
with torch.inference_mode():
 k=torch.randn(1,64,48,128,device='cuda').transpose(1,2);b=torch.rand(1,48,64,device='cuda');e=torch.rand_like(b);w=torch.empty(1,64,48,128,device='cuda');v=torch.empty_like(w);ks,bs,es=k.stride(),b.stride(),e.stride();total=k.numel()
 def ref():return k*b.unsqueeze(-1),(k*b.unsqueeze(-1))*e.unsqueeze(-1)
 planned=_make_key_preparer(k,b,64)
 methods={'torch_three_products':ref,'checked_helper':lambda:prepare_keys(k,b,e),'planned_helper':lambda:planned(k,b,e)}
 for block in [128,256,512,1024,2048]:
  def call(block=block):
   _key_prepare[((total+block-1)//block,)](k,b,e,w,v,total,ks,bs,es,H=48,T=64,D=128,TIME_MAJOR=True,BLOCK=block,num_warps=4,enable_fp_fusion=False)
   return w.transpose(1,2),v.transpose(1,2)
  methods['kernel_block'+str(block)]=call
 expected=ref()
 for label,fn in methods.items():
  for _ in range(4):result=fn()
  assert all(torch.equal(x.contiguous().view(torch.int32),y.contiguous().view(torch.int32)) for x,y in zip(result,expected));del result
  times=[]
  for _ in range(8):
   a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(32):result=fn()
   z.record();z.synchronize();times.append(a.elapsed_time(z)/32);del result
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for _ in range(32):result=fn()
  graph_times=[]
  for _ in range(8):
   a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graph.replay();z.record();z.synchronize();graph_times.append(a.elapsed_time(z)/32)
  d['methods'][label]={'median_eager_cuda_ms':statistics.median(times),'median_graph_cuda_ms':statistics.median(graph_times)};print(label,d['methods'][label],flush=True);del graph,result
 del k,b,e,w,v,expected,methods,fn,ref,a,z
gc.collect();torch.cuda.empty_cache();d['status']='complete';(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
