"""Same resident27B; primary measures are prefill latency and prompt-token throughput."""
from pathlib import Path
import os,sys,json,time,gc,weakref,statistics,importlib.util,textwrap,hashlib

def main():
 import torch
 from transformers import PreTrainedTokenizerFast
 sys.path.insert(0,'/workspace')
 import spark_engine.qwen_model as models
 import spark_engine.qwen_layers as layers
 from spark_engine.qwen_model import QwenCausalLM
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-triton-cache');torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
 source=Path('/harness/mlp_forward_before_activation.py.txt').read_text();ns=dict(vars(models));exec(compile(textwrap.dedent(source),'retained-original-mlp-forward','exec'),ns);original=ns['forward'];candidate=models.QwenMLP.forward
 spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 d={'status':'loading','mode':mode,'scope':'One resident native27B, repeated ABBA prefill only; generation used solely for correctness','primary_metrics':['median_prefill_cuda_ms','median_prefill_wall_ms','prompt_tokens_per_second_cuda'],'triton_cache_dir':os.environ['TRITON_CACHE_DIR'],'original_function_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def cpu(x):
  if torch.is_tensor(x):return x.cpu()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [cpu(v) for v in x]
  return x
 def exact(a,b,path='root'):
  if torch.is_tensor(a):
   if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):d['failure']={'path':path,'actual_shape':list(a.shape),'expected_shape':list(b.shape),'actual_dtype':str(a.dtype),'expected_dtype':str(b.dtype)};save();raise AssertionError('Tensor mismatch at '+path)
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:exact(a[k],b[k],path+'.'+str(k))
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
  else:assert a==b,path
 def select(backend):
  fn=original if backend=='original' else candidate;models.QwenMLP.forward=fn
 save();start=time.perf_counter();model=QwenCausalLM.from_pretrained('/model');torch.cuda.synchronize();d['load_seconds']=time.perf_counter()-start;d['quantization_metadata']=dict(model.quantization_metadata);refs=[(n,weakref.ref(t)) for n,t in list(model.named_parameters())+list(model.named_buffers())];tokenizer=PreTrainedTokenizerFast.from_pretrained('/model',local_files_only=True);base=tokenizer.encode(bench.PROMPT,add_special_tokens=False);d['status']='benchmarking';save();print('MLP_ACTIVATION_AB_MODEL_LOADED',flush=True)
 try:
  with torch.inference_mode():
   for length in [256,1024]:
    canonical=json.loads(Path('/harness/prefill-canonical-generation-v1.json').read_text());canonical_case=next(c for c in canonical['cases'] if c['prompt_tokens']==length);canonical_ids=canonical_case['output_token_ids'];canonical_prompt=canonical_case['prompt_token_ids']
    actual_prompt=(base*((length+len(base)-1)//len(base)))[:length];assert actual_prompt==canonical_prompt,'Prompt identity changed from accepted baseline'
    ids=torch.tensor([actual_prompt],device='cuda');row={'prompt_tokens':length,'prompt_token_ids':actual_prompt,'canonical_generation_reference':'prefill-canonical-generation-v1.json','prefill_repetitions':[],'first_validation_prefill_wall_seconds':{},'generation_validation':[]};d['cases'].append(row);save()
    for backend in ['original','candidate']:
     select(backend);torch.cuda.synchronize();started=time.perf_counter();logits,cache=model.prefill(ids);torch.cuda.synchronize();row['first_validation_prefill_wall_seconds'][backend]=time.perf_counter()-started
     if backend=='original':expected_logits=cpu(logits);expected_cache=cpu(cache)
     else:exact(cpu(logits),expected_logits,'prefill.logits');exact(cpu(cache),expected_cache,'prefill.cache')
     del logits,cache
    del expected_logits,expected_cache;row['prefill_logits_and_all_cache_contents_element_exact']=True;save()
    for backend in ['original','candidate']:
     select(backend);logits,cache=model.prefill(ids);next_id=logits[:,-1,:].argmax(dim=-1,keepdim=True);step_logits,step_cache=model.decode(next_id,cache)
     if backend=='original':expected_logits=cpu(step_logits);expected_cache=cpu(step_cache)
     else:exact(cpu(step_logits),expected_logits,'decode.logits');exact(cpu(step_cache),expected_cache,'decode.cache')
     del logits,cache,next_id,step_logits,step_cache
    del expected_logits,expected_cache;row['next_decode_logits_and_all_cache_contents_element_exact']=True
    for backend in ['original','candidate','candidate','original']*8:
     select(backend);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();logits,cache=model.prefill(ids);b.record();b.synchronize();wall_ms=(time.perf_counter()-start)*1000;row['prefill_repetitions'].append({'backend':backend,'cuda_ms':a.elapsed_time(b),'wall_ms':wall_ms,'peak_allocated_bytes':torch.cuda.max_memory_allocated()});del logits,cache
    row['summary']={backend:{'median_prefill_cuda_ms':statistics.median(x['cuda_ms'] for x in row['prefill_repetitions'] if x['backend']==backend),'median_prefill_wall_ms':statistics.median(x['wall_ms'] for x in row['prefill_repetitions'] if x['backend']==backend)} for backend in ['original','candidate']}
    for backend in row['summary']:row['summary'][backend]['prompt_tokens_per_second_cuda']=length*1000/row['summary'][backend]['median_prefill_cuda_ms']
    row['prefill_speedup']=row['summary']['original']['median_prefill_cuda_ms']/row['summary']['candidate']['median_prefill_cuda_ms'];row['less_prefill_time_pct']=100*(1-1/row['prefill_speedup']);save()
    expected_tokens=canonical_ids
    for backend in ['original','candidate']*2:
     select(backend)
     # Diagnostic copies happen after each GPU argmax, without CPU synchronization.
     # Only correctness generations are instrumented; primary prefill timing is untouched.
     trace=torch.empty((128,248320),device='cuda',dtype=torch.bfloat16);argmax_impl=torch.Tensor.argmax;trace_step=[0]
     def capture_argmax(t,*args,**kwargs):
      result=argmax_impl(t,*args,**kwargs)
      if t.shape==(1,248320):trace[trace_step[0]].copy_(t[0]);trace_step[0]+=1
      return result
     try:
      torch.Tensor.argmax=capture_argmax;generated=model.generate(ids,128,None)
     finally:torch.Tensor.argmax=argmax_impl
     tokens=generated[0,length:].tolist();assert len(tokens)==128 and trace_step[0]==128
     host_trace=trace.cpu();cpu_argmax=argmax_impl(host_trace,-1).tolist();nonfinite=int((~torch.isfinite(host_trace)).sum());argmax_matches=cpu_argmax==tokens
     if length==256:
      vals,inds=host_trace[5].float().topk(5);row.setdefault('outlier_step5_observations',[]).append({'backend':backend,'top5_ids':inds.tolist(),'top5_logits':vals.tolist(),'historical_wrong_token':78326,'wrong_token_logit':float(host_trace[5,78326]),'canonical_token':9989,'canonical_token_logit':float(host_trace[5,9989])})
     if nonfinite or not argmax_matches or tokens!=canonical_ids:
      torch.save({'logits':host_trace,'gpu_token_ids':tokens,'cpu_argmax_ids':cpu_argmax,'canonical_ids':canonical_ids},out/(str(length)+'-'+backend+'-generation-'+str(len(row['generation_validation']))+'-trace.pt'))
     assert nonfinite==0 and argmax_matches,'Nonfinite logits or GPU/CPU argmax mismatch; trace retained'
     del trace,host_trace,capture_argmax
     if expected_tokens is None:expected_tokens=tokens
     elif tokens!=expected_tokens:d['failure']={'classification':'accepted-baseline generation mismatch','prompt_tokens':length,'backend':backend,'expected_token_ids':expected_tokens,'actual_token_ids':tokens,'first_mismatch':next(i for i,(x,y) in enumerate(zip(expected_tokens,tokens)) if x!=y)};save();raise AssertionError('128 generation-validation token IDs changed')
     row['generation_validation'].append({'backend':backend,'output_token_ids':tokens,'finite_logits':True,'gpu_cpu_argmax_exact':True,'logits_capture':'GPU copy after argmax, no per-token CPU sync'});del generated
    row['generation_ids_element_exact']=True;row['generation_repeats_per_backend']=2
    for backend in ['original','candidate']:select(backend);row[backend+'_first_token_cuda_diagnostic']=bench.timed_diagnostic(torch,model,ids,1)
    save();print(json.dumps({'prompt_tokens':length,'prefill':row['summary'],'less_prefill_time_pct':row['less_prefill_time_pct'],'generation_ids_exact':True}),flush=True);del ids
 finally:select('candidate')
 del model;gc.collect();d['live_model_tensors_after_release']=[n for n,r in refs if r() is not None];assert not d['live_model_tensors_after_release'];d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;d['status']='complete';save();print('MLP_ACTIVATION_AB_COMPLETE',flush=True)
if __name__=='__main__':main()
