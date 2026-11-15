"""Reference-only offline vLLM benchmark; no native engine implementation."""
import os,time,json,hashlib,gc,statistics
from pathlib import Path

def main():
 import torch,vllm
 from vllm import LLM,SamplingParams
 from transformers import AutoTokenizer
 out=Path('/results');model=Path('/model')
 def save(value):
  tmp=out/'results.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out/'results.json')
 def phase(label,**more):
  with (out/'phases.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':label,**more})+'\n');f.flush();os.fsync(f.fileno())
 config=dict(model=str(model),tokenizer=str(model),trust_remote_code=False,dtype='bfloat16',max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=.40,kv_cache_memory_bytes=512*1024**2,enable_prefix_caching=False,enforce_eager=True,language_model_only=True,seed=0,disable_log_stats=False)
 d={'status':'loading','runtime':'vllm','vllm_version':vllm.__version__,'torch_version':torch.__version__,'settings':config,'cases':[],'speculative_decoding':False,'cuda_graphs':False,'measurement':'Host wall completion latency, excludes load/tokenization; vLLM internal metrics reported separately'};save(d)
 phase('before_load');start=time.perf_counter();llm=LLM(**config);d['load_ms']=(time.perf_counter()-start)*1000;d['status']='benchmarking';save(d);phase('after_load',load_ms=d['load_ms']);print('REFERENCE_LOADED',flush=True)
 tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 import ast
 tree=ast.parse(Path('/workspace/tools/benchmark_qwen.py').read_text());prompt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in n.targets));base=tokenizer.encode(prompt,add_special_tokens=False)
 params=SamplingParams(temperature=0,max_tokens=128,min_tokens=128,ignore_eos=True,seed=0)
 for length in (256,1024):
  ids=(base*((length+len(base)-1)//len(base)))[:length];case={'prompt_tokens':length,'output_tokens':128,'prompt_token_ids':ids,'samples':[]}
  for rep in (-1,0,1,2):
   phase('before_generation',prompt_tokens=length,repetition=rep);start=time.perf_counter();results=llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False);elapsed=(time.perf_counter()-start)*1000;r=results[0];tokens=list(r.outputs[0].token_ids);assert len(tokens)==128 and len(r.prompt_token_ids)==length
   row={'repetition':rep,'wall_ms':elapsed,'output_tokens_per_second':128000/elapsed,'output_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'output_token_ids':tokens,'text':r.outputs[0].text}
   if r.metrics is not None:row['vllm_metrics']={k:v for k,v in vars(r.metrics).items() if isinstance(v,(float,int,str,type(None)))}
   phase('after_generation',prompt_tokens=length,repetition=rep,wall_ms=elapsed,output_sha256=row['output_sha256']);print(json.dumps({'prompt_tokens':length,'repetition':rep,'wall_ms':elapsed,'output_tok_s':row['output_tokens_per_second']}),flush=True)
   if rep>=0:case['samples'].append(row);save({**d,'current_case':case})
  assert len({x['output_sha256'] for x in case['samples']})==1
  median=statistics.median(x['wall_ms'] for x in case['samples']);case['summary']={'median_wall_ms':median,'median_output_tokens_per_second':128000/median,'outputs_match':True};d['cases'].append(case);save(d)
 d['status']='complete';save(d);phase('before_release');del llm,results,r;gc.collect();torch.cuda.empty_cache();phase('worker_exiting');print('REFERENCE_BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
