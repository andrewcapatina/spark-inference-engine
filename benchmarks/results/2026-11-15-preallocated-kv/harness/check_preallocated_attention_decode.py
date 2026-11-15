"""Old/new attention decode on shared tiny weights; exact state and counted KV copies."""
from pathlib import Path
import sys,json,gc,tempfile,textwrap,importlib.util
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_layers as layers
 import test_quantized_loader as fixture
 from spark_engine.qwen_quantized import load_quantized_qwen
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(22022);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 ns=dict(vars(layers));source=Path('/harness/qwen_attention_decode_before_preallocation.py.txt').read_text();exec(compile(textwrap.dedent(source),'retained-original-attention-decode','exec'),ns);original=ns['decode'];d={'status':'running','mode':mode,'scope':'Tiny dense/mixedquant attention blocks, original/current method on same weights; no real-checkpoint claim','cases':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def equal(a,ca,b,cb):
  assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Decode output changed'
  for x,y in zip(ca,cb):assert x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y),'KV values/layout shape changed'
 def counted(fn,hidden,pos,cache):
  previous=torch.cat;counter=[0]
  def cat(tensors,*args,**kwargs):
   dim=kwargs.get('dim',args[0] if args else 0)
   if dim==2 and all(t.ndim==4 for t in tensors):counter[0]+=1
   return previous(tensors,*args,**kwargs)
  torch.cat=cat
  try:result=fn(hidden,pos,cache)
  finally:torch.cat=previous
  return result,counter[0]
 def cases(block,kind,dtype,batches,lengths):
  for batch in batches:
   for length in lengths:
    total=length+3;raw=torch.randn(batch,total,block.q_proj.in_features*2 if isinstance(block.q_proj,torch.nn.Linear) else 128*2,device='cuda',dtype=dtype);hidden=raw[...,::2];phase=torch.randn(1,total,4,device='cuda',dtype=torch.float32);cos,sin=phase.cos().to(dtype),phase.sin().to(dtype);mask=torch.triu(torch.full((length,length),float('-inf'),device='cuda',dtype=dtype),diagonal=1)
    _,initial=block.prefill(hidden[:,:length],(cos[:,:length],sin[:,:length]),mask);prefix=tuple(t.clone() for t in initial);old_cache=new_cache=initial
    for step in range(3):
     position=length+step;item=hidden[:,position:position+1];pos=(cos[:,position:position+1],sin[:,position:position+1]);(a,old_cache),old_count=counted(lambda *args:original(block,*args),item,pos,old_cache);(b,new_cache),new_count=counted(block.decode,item,pos,new_cache);assert old_count==2 and new_count==0;equal(a,old_cache,b,new_cache);assert new_cache[0].shape==(batch,block.num_kv_heads,position+1,block.head_dim)
    for x,y in zip(initial,prefix):assert torch.equal(x,y),'Input prefix mutated'
    snapshot=tuple(t.clone() for t in new_cache);branch_hidden=-hidden[:,length:length+1];branch_pos=(cos[:,length:length+1],sin[:,length:length+1]);a,ca=original(block,branch_hidden,branch_pos,initial);b,cb=block.decode(branch_hidden,branch_pos,initial);equal(a,ca,b,cb)
    for x,y in zip(new_cache,snapshot):assert torch.equal(x,y),'Branch mutated retained cache'
    d['cases'].append({'weights':kind,'dtype':str(dtype),'batch':batch,'prefix_length':length,'kv_groups':block.num_groups,'exact_three_steps_and_branch':True,'kv_cat_calls_before':2,'kv_cat_calls_after':0});save()
 save()
 with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:
  for dtype in [torch.float32,torch.bfloat16]:
   for kvheads in [1,2,4]:
    cfg={'hidden_size':64,'num_attention_heads':4,'num_key_value_heads':kvheads,'head_dim':16};block=layers.QwenAttention(cfg).to('cuda',dtype=dtype).eval();cases(block,'dense',dtype,[1,2],[1,3,4,63,64,65,255,256]);del block;gc.collect();torch.cuda.empty_cache()
  fixture.create_fixture(Path(tmp));gc.collect();torch.cuda.empty_cache();model=load_quantized_qwen(tmp);block=model.layers[0].self_attn;cases(block,'mixed_fp8_nvfp4',torch.bfloat16,[1,2],[1,5,63,64,65,255,256]);del block,model
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2
 d['status']='complete';save();print('PREALLOCATED_ATTENTION_DECODE_EXACT_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
