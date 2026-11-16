"""Compare actual batched preparation intermediates with individual chunk calls."""
from pathlib import Path
import ast,inspect,json,sys
sys.path.insert(0,'/workspace')
import torch
import spark_engine.qwen_ops as ops
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
tree=ast.parse(inspect.getsource(ops._prepare_gdn_chunks));tree.body[0].body[-1]=ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]));ast.fix_missing_locations(tree);ns=dict(vars(ops));exec(compile(tree,'observed-original-preparation','exec'),ns);observe=ns['_prepare_gdn_chunks'];original=ops._prepare_gdn_chunks;rows=[]
def wrapped(*args):
 result=original(*args)
 if args[0].ndim!=5:return result
 batched=observe(*args);case=[]
 for i in range(args[0].shape[2]):
  single=observe(*(x.select(2,i) for x in args));differences=[]
  for name in ['cum_decay','pairwise_decay','v_beta','decayed_k','ut_system','intra_chunk_attn','v_new','k_cumdecay']:
   a,b=single[name],batched[name].select(2,i)
   differences.append({'name':name,'exact':torch.equal(a,b),'max_abs':float((a-b).abs().nan_to_num().max()),'different_elements':int((a!=b).sum()),'single_stride':list(a.stride()),'batch_slice_stride':list(b.stride())})
  case.append({'chunk':i,'operations':differences})
 rows.append(case);return result
ops._prepare_gdn_chunks=wrapped
with torch.inference_mode():
 for heads,dim in [(3,16),(48,128)]:
  for length in [129,256,1024]:
   torch.manual_seed(8192+length)
   def rand():return torch.randn(1,length,heads,dim*2,device='cuda',dtype=torch.bfloat16)[...,::2]
   args=(rand(),rand(),rand(),-torch.rand(1,length,heads,device='cuda',dtype=torch.bfloat16)*.1,torch.rand(1,length,heads,device='cuda',dtype=torch.bfloat16));ops.gated_delta_chunk(*args)
with torch.inference_mode():
 import spark_engine.qwen_layers as layers
 for heads in [3,48]:
  block=layers.QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':1 if heads==3 else 16,'linear_num_value_heads':heads,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval()
  for length in [65,256,1024]:
   hidden=torch.randn(1,length,256,device='cuda',dtype=torch.bfloat16)[...,::2];block.prefill(hidden)
  del block,hidden
ops._prepare_gdn_chunks=original
d={'status':'complete','mode':mode,'scope':'Synthetic actual preparation shapes, original helper operations observationally captured; no real checkpoint','calls':rows};(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
summary={}
for call in rows:
 for c in call:
  for x in c['operations']:
   if not x['exact']:summary[x['name']]=max(summary.get(x['name'],0),x['max_abs'])
print(json.dumps(summary),flush=True)
