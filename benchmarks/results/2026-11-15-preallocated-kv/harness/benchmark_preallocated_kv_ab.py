"""Single resident native model; retained original versus local-Qwen preallocated attention KV cache."""
from pathlib import Path
import os,sys,json,time,gc,weakref,statistics,importlib.util,textwrap,types,hashlib

def main():
 import torch
 from transformers import PreTrainedTokenizerFast
 sys.path.insert(0,'/workspace')
 import spark_engine.qwen_layers as layers
 from spark_engine.qwen_model import QwenCausalLM
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ.pop('SPARK_GDN_PREFILL_BACKEND',None)
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
 source=Path('/harness/qwen_attention_decode_before_preallocation.py.txt').read_text();namespace=dict(vars(layers));exec(compile(textwrap.dedent(source),'retained-original-prefill','exec'),namespace);original=namespace['decode'];candidate=layers.QwenAttention.decode
 spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 d={'status':'loading','mode':mode,'scope':'One resident27B; original versus preallocated attention KV cache only; exact outputs/caches before A/B timings','original_method_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':[]}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def cpu(x):
  if torch.is_tensor(x):return x.cpu()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [cpu(v) for v in x]
  return x
 def exact(a,b):
  if torch.is_tensor(a):assert torch.equal(a,b),'Tensor mismatch'
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:exact(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):exact(x,y)
  else:assert a==b
 save();start=time.perf_counter();model=QwenCausalLM.from_pretrained('/model');torch.cuda.synchronize();d['load_seconds']=time.perf_counter()-start;d['quantization_metadata']=dict(model.quantization_metadata);refs=[(n,weakref.ref(t)) for n,t in list(model.named_parameters())+list(model.named_buffers())];mixers=[m for m in model.modules() if isinstance(m,layers.QwenAttention)];assert len(mixers)==16
 def select(backend):
  for mixer in mixers:mixer.decode=types.MethodType(original if backend=='original' else candidate,mixer)
 tokenizer=PreTrainedTokenizerFast.from_pretrained('/model',local_files_only=True);base=tokenizer.encode(bench.PROMPT,add_special_tokens=False);d['status']='benchmarking';save();print('PREALLOCATED_KV_AB_MODEL_LOADED',flush=True)
 with torch.inference_mode():
  for length in [256,1024]:
   ids=torch.tensor([(base*((length+len(base)-1)//len(base)))[:length]],device='cuda');row={'prompt_tokens':length,'prefill_repetitions':[],'generation_repetitions':[],'decode_scope':'preallocated cache versus accepted two-concatenation baseline'}
   for backend in ['original','reuse']:
    select(backend);logits,cache=model.prefill(ids)
    if backend=='original':expected_logits=cpu(logits);expected_cache=cpu(cache)
    else:exact(cpu(logits),expected_logits);exact(cpu(cache),expected_cache)
    del logits,cache
   del expected_logits,expected_cache;row['logits_and_all_cache_contents_element_exact']=True
   for backend in ['original','reuse']:
    select(backend);logits,cache=model.prefill(ids);next_id=logits[:,-1,:].argmax(dim=-1,keepdim=True);step_logits,step_cache=model.decode(next_id,cache)
    if backend=='original':expected_decode_logits=cpu(step_logits);expected_decode_cache=cpu(step_cache)
    else:exact(cpu(step_logits),expected_decode_logits);exact(cpu(step_cache),expected_decode_cache)
    del logits,cache,next_id,step_logits,step_cache
   del expected_decode_logits,expected_decode_cache;row['decode_logits_and_all_cache_contents_element_exact']=True
   for backend in ['original','reuse','reuse','original']:
    select(backend);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();logits,cache=model.prefill(ids);b.record();b.synchronize();wall=(time.perf_counter()-start)*1000
    row['prefill_repetitions'].append({'backend':backend,'cuda_ms':a.elapsed_time(b),'wall_ms':wall,'peak_allocated_bytes':torch.cuda.max_memory_allocated()});del logits,cache
   expected_tokens=None
   for backend in ['original','reuse','reuse','original']:
    select(backend);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();generated=model.generate(ids,128,None);torch.cuda.synchronize();wall=time.perf_counter()-start;tokens=generated[0,length:].tolist();assert len(tokens)==128
    if expected_tokens is None:expected_tokens=tokens
    elif tokens!=expected_tokens:
     d['failure']={'prompt_tokens':length,'backend':backend,'expected_token_ids':expected_tokens,'actual_token_ids':tokens,'first_mismatch':next(i for i,(x,y) in enumerate(zip(expected_tokens,tokens)) if x!=y)};save();raise AssertionError('128 generated token IDs changed')
    row['generation_repetitions'].append({'backend':backend,'seconds':wall,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'total_output_tokens_per_second':128/wall,'output_token_ids':tokens});del generated
   for backend in ['original','reuse']:
    select(backend);row[backend+'_cuda_diagnostic']=bench.timed_diagnostic(torch,model,ids,128);row[backend+'_median_prefill_cuda_ms']=statistics.median(x['cuda_ms'] for x in row['prefill_repetitions'] if x['backend']==backend)
   row['prefill_speedup']=row['original_median_prefill_cuda_ms']/row['reuse_median_prefill_cuda_ms'];d['cases'].append(row);save();print(json.dumps({'length':length,'prefill_speedup':row['prefill_speedup'],'exact':True}),flush=True);del ids
 for m in mixers:delattr(m,'decode')
 del mixers,m,model;gc.collect();d['live_model_tensors_after_release']=[n for n,r in refs if r() is not None];assert not d['live_model_tensors_after_release'];d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2
 d['status']='complete';save();print('PREALLOCATED_KV_AB_COMPLETE',flush=True)
if __name__=='__main__':main()
