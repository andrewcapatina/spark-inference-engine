"""Same resident model: baseline repeatability, cache layouts, paired teacher-forced decode."""
from pathlib import Path
import sys,json,gc,time,weakref,importlib.util
sys.path.insert(0,'/workspace')
def main():
 import torch
 from transformers import PreTrainedTokenizerFast
 import spark_engine.qwen_ops as ops
 from spark_engine.qwen_model import QwenCausalLM
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
 ns=dict(vars(ops));exec(compile(Path('/harness/qwen_ops_before_state_processing.py.txt').read_text(),'retained-state-baseline','exec'),ns);original=ns['gated_delta_chunk'];candidate=ops.gated_delta_chunk
 spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 d={'status':'loading','mode':mode,'scope':'Diagnostic only: original-repeat versus candidate, one resident checkpoint; no accepted performance/correctness claim','generation_runs':[],'teacher_pairs':[]};
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def select(label):ops.gated_delta_chunk=original if label=='original' else candidate
 def tensors(value,path='cache'):
  if torch.is_tensor(value):yield path,value
  elif isinstance(value,dict):
   for k,v in value.items():yield from tensors(v,path+'.'+str(k))
  elif isinstance(value,(tuple,list)):
   for i,v in enumerate(value):yield from tensors(v,path+'.'+str(i))
 def layout(value):return {k:{'shape':list(v.shape),'stride':list(v.stride()),'offset':v.storage_offset(),'dtype':str(v.dtype),'contiguous':v.is_contiguous()} for k,v in tensors(value)}
 def compare(a,b):
  aa=dict(tensors(a));bb=dict(tensors(b));assert aa.keys()==bb.keys();different=[]
  for key,x in aa.items():
   y=bb[key]
   if x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y):
    delta=(x.float()-y.float()).abs();different.append({'path':key,'shape':list(x.shape),'max_abs':delta.max().item(),'different_elements':int(torch.count_nonzero(x!=y).item()),'a_stride':list(x.stride()),'b_stride':list(y.stride())})
  return different
 save();model=QwenCausalLM.from_pretrained('/model');refs=[(n,weakref.ref(v)) for n,v in list(model.named_parameters())+list(model.named_buffers())];tokenizer=PreTrainedTokenizerFast.from_pretrained('/model',local_files_only=True);base=tokenizer.encode(bench.PROMPT,add_special_tokens=False);length=256;ids=torch.tensor([(base*((length+len(base)-1)//len(base)))[:length]],device='cuda');d['status']='diagnosing';save()
 try:
  with torch.inference_mode():
   for label in ['original','original','candidate','candidate']:
    select(label);generated=model.generate(ids,128,None);tokens=generated[0,length:].tolist();assert len(tokens)==128;d['generation_runs'].append({'backend':label,'output_token_ids':tokens,'matches_first':tokens==d['generation_runs'][0]['output_token_ids'] if d['generation_runs'] else True});del generated;save();print('GENERATION_COMPLETE '+label,flush=True)
   for left,right in [('original','original'),('original','candidate')]:
    select(left);la,ca=model.prefill(ids.clone());select(right);lb,cb=model.prefill(ids.clone());row={'backends':[left,right],'prefill_logit_differences':compare(la,lb),'prefill_cache_differences':compare(ca,cb),'a_layout':layout(ca),'b_layout':layout(cb),'decode_steps':[]};d['teacher_pairs'].append(row);save()
    for step in range(127):
     token=la[:,-1,:].argmax(-1,keepdim=True);select(left);la,ca=model.decode(token,ca);select(right);lb,cb=model.decode(token,cb);logit_diff=compare(la,lb);cache_diff=compare(ca,cb);record={'step':step+1,'logit_differences':logit_diff,'cache_differences':cache_diff,'greedy_ids_same':torch.equal(la[:,-1,:].argmax(-1),lb[:,-1,:].argmax(-1))};row['decode_steps'].append(record)
     if logit_diff or cache_diff:
      row['first_difference_step']=step+1;row['a_layout_at_difference']=layout(ca);row['b_layout_at_difference']=layout(cb);path=(cache_diff or logit_diff)[0]['path'];values_a=dict(tensors(ca if cache_diff else la));values_b=dict(tensors(cb if cache_diff else lb));torch.save({'a':values_a[path].cpu(),'b':values_b[path].cpu(),'step':step+1,'path':path},out/('first-difference-'+left+'-'+right+'.pt'));del values_a,values_b,path;save();break
     if step%16==0:save()
    del la,lb,ca,cb,token;save();print('TEACHER_PAIR_COMPLETE '+left+' '+right,flush=True)
 finally:select('candidate')
 del model,ids;gc.collect();d['live_model_tensors_after_release']=[n for n,v in refs if v() is not None];assert not d['live_model_tensors_after_release'];d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;d['status']='complete';save();print('STATE_GENERATION_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
