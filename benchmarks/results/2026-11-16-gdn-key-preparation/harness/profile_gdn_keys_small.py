"""Measured actual convolution-produced layout; no real checkpoint load."""
from pathlib import Path
import sys,json,gc,statistics,ast
sys.path.insert(0,'/workspace');sys.path.insert(0,'/harness')
import torch
from torch.profiler import profile,ProfilerActivity,record_function
import spark_engine.qwen_ops as ops
from spark_engine.qwen_layers import QwenGatedDeltaNet
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
ns=dict(vars(ops));exec(compile(Path('/harness/qwen_ops_before_keys_processing.py.txt').read_text(),'accepted-baseline','exec'),ns);original=ns['gated_delta_chunk'];candidate=ops.gated_delta_chunk
d={'status':'running','mode':mode,'scope':'Synthetic block with actual native48-head convolution layouts; timings not checkpoint results','cases':[]}
with torch.inference_mode():
 block=QwenGatedDeltaNet({'hidden_size':128,'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}).to('cuda',dtype=torch.bfloat16).eval()
 for length in [256,1024]:
  captured=[]
  def capture(*a,**kw):captured.append(a);return original(*a,**kw)
  ops.gated_delta_chunk=capture;hidden=torch.randn(1,length,128,device='cuda',dtype=torch.bfloat16);result=block.prefill(hidden);ops.gated_delta_chunk=candidate;assert len(captured)==1;args=captured.pop();del result,hidden
  row={'length':length,'inputs':[{'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype)} for x in args if torch.is_tensor(x)],'samples':[],'profiles':{}}
  for fn in [original,candidate]:
   for _ in range(3):result=fn(*args);del result
  for label in ['original','candidate','candidate','original']*4:
   fn=original if label=='original' else candidate;a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();result=fn(*args);b.record();b.synchronize();row['samples'].append({'backend':label,'cuda_ms':a.elapsed_time(b)});del result
  row['medians']={b:statistics.median(x['cuda_ms'] for x in row['samples'] if x['backend']==b) for b in ['original','candidate']}
  for label,fn in [('original',original),('candidate',candidate)]:
   with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
    with record_function('native.engine.prefill'),record_function('native.gdn.state_update'):result=fn(*args);torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/f'{length}-{label}-trace.json'));row['profiles'][label]=[{'name':x.key,'calls':x.count,'self_gpu_ms':x.self_device_time_total/1000,'total_gpu_ms':x.device_time_total/1000,'shapes':x.input_shapes} for x in prof.key_averages(group_by_input_shape=True) if x.device_time_total];del result,prof
  d['cases'].append(row);print(json.dumps({'length':length,'medians':row['medians']}),flush=True);del args
 del block
gc.collect();torch.cuda.empty_cache();d['status']='complete';(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
