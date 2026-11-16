"""Strict small-tensor sweep of bounded batch sizes on actual convolution layout."""
from pathlib import Path
import sys,json,ast,inspect,statistics,gc
sys.path.insert(0,'/workspace')
import torch
import spark_engine.qwen_ops as ops
from spark_engine.qwen_layers import QwenGatedDeltaNet
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
ns=dict(vars(ops));exec(compile(Path('/harness/qwen_ops_before_batch_processing.py.txt').read_text(),'baseline','exec'),ns);original=ns['gated_delta_chunk'];variants={'original':original}
for limit in [2,4,8,16]:
 tree=ast.parse(inspect.getsource(ops.gated_delta_chunk))
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='batch_chunks' for t in n.targets):
   n.value.body.args[0]=ast.Constant(value=limit)
   if limit==16:
    for x in ast.walk(n.value):
     if isinstance(x,ast.Constant) and x.value==256:x.value=512
 ast.fix_missing_locations(tree);scope=dict(vars(ops));exec(compile(tree,'group-'+str(limit),'exec'),scope);variants['group'+str(limit)]=scope['gated_delta_chunk']
d={'status':'running','mode':mode,'scope':'Synthetic native48-head convolution layout only, each size output/state exact vs accepted; group16 uses512MiB estimate','cases':[]}
with torch.inference_mode():
 block=QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval()
 for length in [256,1024]:
  captured=[]
  def capture(*a,**kw):captured.append(a);return original(*a,**kw)
  ops.gated_delta_chunk=capture;result=block.prefill(torch.randn(1,length,128,device='cuda',dtype=torch.bfloat16));args=captured.pop();del result;ops.gated_delta_chunk=variants['group8'];expected=original(*args)
  row={'length':length,'samples':[]}
  for label,fn in variants.items():
   actual=fn(*args);assert all(a.dtype==b.dtype and torch.equal(a,b) for a,b in zip(expected,actual)),label;del actual
   for _ in range(3):result=fn(*args);del result
  order=list(variants);order=order+order[::-1]
  for label in order*4:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();result=variants[label](*args);b.record();b.synchronize();row['samples'].append({'backend':label,'cuda_ms':a.elapsed_time(b)});del result
  row['medians']={label:statistics.median(x['cuda_ms'] for x in row['samples'] if x['backend']==label) for label in variants};d['cases'].append(row);print(json.dumps({'length':length,'medians':row['medians']}),flush=True);del args,expected
 del block
gc.collect();torch.cuda.empty_cache();d['status']='complete';(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
