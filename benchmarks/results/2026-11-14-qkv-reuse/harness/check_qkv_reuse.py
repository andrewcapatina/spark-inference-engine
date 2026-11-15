"""Exact old/new GDN-prefill comparison on one set of dense or quantized weights."""
from pathlib import Path
import sys,os,json,gc,tempfile,textwrap,importlib.util
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
def main():
 import torch
 import spark_engine.qwen_layers as layers
 import test_quantized_loader as fixture
 from spark_engine.qwen_quantized import load_quantized_qwen
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ.pop('SPARK_GDN_PREFILL_BACKEND',None)
 torch.set_num_threads(4);torch.manual_seed(22020);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 source=Path('/harness/qwen_gdn_prefill_before_storage.py.txt').read_text();namespace=dict(vars(layers));exec(compile(textwrap.dedent(source),'retained-original-prefill','exec'),namespace);original=namespace['prefill']
 d={'status':'running','mode':mode,'scope':'Single shared block weights; exact original/new output, cache and continuation; counted QKV calls; tiny models only','cases':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def equivalent(a,ca,b,cb):
  assert torch.equal(a,b),'Output changed';assert ca.keys()==cb.keys()
  for key in ca:
   if torch.is_tensor(ca[key]):assert torch.equal(ca[key],cb[key]),'Cache changed: '+key
   else:assert ca[key]==cb[key]
 save()
 with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:
  for kind in ['dense','mixed_fp8_nvfp4']:
   if kind=='dense':
    cfg={'hidden_size':128,'linear_num_key_heads':1,'linear_num_value_heads':3,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4};block=layers.QwenGatedDeltaNet(cfg).to('cuda',dtype=torch.bfloat16).eval()
   else:
    fixture.create_fixture(Path(tmp));gc.collect();torch.cuda.empty_cache();model=load_quantized_qwen(tmp);block=model.layers[1].linear_attn
   for batch in [1,2]:
    for length in [1,3,4,5,64,256,1024]:
     hidden=torch.randn(batch,length,128,device='cuda',dtype=torch.bfloat16);before=hidden.clone();counter=[0]
     def count(*args):counter[0]+=1
     hook=block.in_proj_qkv.register_forward_hook(count)
     expected,old_cache=original(block,hidden);assert counter[0]==2
     counter[0]=0;actual,new_cache=block.prefill(hidden);assert counter[0]==1;hook.remove();equivalent(expected,old_cache,actual,new_cache);assert torch.equal(hidden,before)
     next_hidden=torch.randn(batch,1,128,device='cuda',dtype=torch.bfloat16);a,ca=block.decode(next_hidden,old_cache);b,cb=block.decode(next_hidden,new_cache);equivalent(a,ca,b,cb)
     row={'weights':kind,'batch':batch,'length':length,'output_cache_continuation_element_exact':True,'qkv_calls_before':2,'qkv_calls_after':1};d['cases'].append(row);save();del hidden,before,expected,actual,old_cache,new_cache,next_hidden,a,b,ca,cb,hook
   del block
   if kind!='dense':del model
   gc.collect();torch.cuda.empty_cache()
 spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2
 d['status']='complete';save();print('QKV_REUSE_EXACT_CHECKS_PASSED '+str(len(d['cases'])),flush=True)
if __name__=='__main__':main()
