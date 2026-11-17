"""Accuracy/likelihood harness only. No model implementation or engine edits."""
from pathlib import Path
import sys,json,time,hashlib,math,gc,weakref,importlib.util,os

def main():
    import torch
    from transformers import PreTrainedTokenizerFast
    sys.path.insert(0,'/workspace');sys.path.insert(0,'/workspace/tools')
    from accuracy_scoring import grade
    run,mode,suite_sha=sys.argv[1:4];out=Path('/results')/run;out.mkdir(exist_ok=True)
    path=Path('/workspace/benchmarks/accuracy/nvfp4-smoke-v1.json');assert hashlib.sha256(path.read_bytes()).hexdigest()==suite_sha;suite=json.loads(path.read_text());assert len(suite['cases'])==40 and len(suite['corpus'])==4
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=PreTrainedTokenizerFast.from_pretrained('/model',local_files_only=True);eos=tokenizer.eos_token_id
    prompts={c['id']:tokenizer.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for c in suite['cases']}
    docs={c['id']:tokenizer.encode(c['text'],add_special_tokens=False) for c in suite['corpus']}
    assert all(len(prompts[c['id']])+c['max_new_tokens']<=2048 for c in suite['cases'])
    d={'status':'loading','mode':mode,'scope':suite['scope'],'suite_sha256':suite_sha,'scoring_sha256':hashlib.sha256(Path('/workspace/tools/accuracy_scoring.py').read_bytes()).hexdigest(),'criteria':suite['criteria_fixed_before_outputs'],'cases':[],'corpus':[],'eos_token_id':eos,'max_prompt_tokens':max(map(len,prompts.values())),'temperature':0,'thinking':False,'native_eos_policy':'single tokenizer EOS; reference finish reason retained','reference_kv_dtype':'bfloat16'}
    def save():
        p=out/'results.tmp'
        with p.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        p.replace(out/'results.json')
    save();start=time.perf_counter()
    if mode=='native':
        from spark_engine.qwen_model import QwenCausalLM
        torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
        model=QwenCausalLM.from_pretrained('/model');model.eval();refs=[(n,weakref.ref(t)) for n,t in list(model.named_parameters())+list(model.named_buffers())]
    elif mode=='reference':
        import vllm
        from vllm import LLM,SamplingParams
        config=dict(model='/model',tokenizer='/model',trust_remote_code=False,dtype='bfloat16',max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=.40,kv_cache_memory_bytes=512*1024**2,enable_prefix_caching=False,kv_cache_dtype='bfloat16',enforce_eager=True,language_model_only=True,seed=0)
        d['reference_settings']=config;d['vllm_version']=vllm.__version__;llm=LLM(**config)
    else:raise ValueError(mode)
    d['load_seconds_diagnostic_only']=time.perf_counter()-start;d['status']='evaluating';save();print('ACCURACY_MODEL_LOADED '+mode,flush=True)
    with torch.inference_mode():
        for case in suite['cases']:
            values=prompts[case['id']]
            if mode=='native':
                ids=torch.tensor([values],device='cuda');generated=model.generate(ids,max_new_tokens=case['max_new_tokens'],eos_token_id=eos);assert torch.equal(generated[:,:len(values)],ids)
                tokens=generated[0,len(values):].tolist();stop='stop' if eos in tokens else 'length';answer=tokenizer.decode(tokens,skip_special_tokens=True);del ids,generated
            else:
                output=llm.generate([{'prompt_token_ids':values}],SamplingParams(temperature=0,max_tokens=case['max_new_tokens'],stop_token_ids=[eos],seed=0),use_tqdm=False)[0];part=output.outputs[0];tokens=list(part.token_ids);answer=tokenizer.decode(tokens,skip_special_tokens=True);stop=part.finish_reason;del output,part
            row={'id':case['id'],'category':case['category'],'prompt_token_ids':values,'output_token_ids':tokens,'reply':answer,'finish_reason':stop,'truncated':stop=='length',**grade(case,answer)};d['cases'].append(row);save();print(json.dumps({'case':case['id'],'passed':row['passed'],'truncated':row['truncated'],'reply':answer}),flush=True)
        for document in suite['corpus']:
            values=docs[document['id']];assert len(values)>1
            if mode=='native':
                ids=torch.tensor([values],device='cuda');logits,cache=model.prefill(ids)
                losses=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none');nll=losses.cpu().tolist();del ids,logits,cache,losses
            else:
                output=llm.generate([{'prompt_token_ids':values}],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=0),use_tqdm=False)[0]
                assert output.prompt_token_ids==values
                nll=[-output.prompt_logprobs[i][values[i]].logprob for i in range(1,len(values))];del output
            assert len(nll)==len(values)-1 and all(math.isfinite(x) and x>=0 for x in nll)
            row={'id':document['id'],'token_ids':values,'scored_tokens':len(nll),'sum_nll':sum(nll),'token_nll':nll,'perplexity':math.exp(sum(nll)/len(nll))};d['corpus'].append(row);save();print(json.dumps({'document':row['id'],'scored_tokens':len(nll),'perplexity':row['perplexity']}),flush=True)
    d['summary']={'passed':sum(c['passed'] for c in d['cases']),'total':len(d['cases']),'truncated':sum(c['truncated'] for c in d['cases']),'scored_tokens':sum(c['scored_tokens'] for c in d['corpus']),'sum_nll':sum(c['sum_nll'] for c in d['corpus'])};d['summary']['perplexity']=math.exp(d['summary']['sum_nll']/d['summary']['scored_tokens']);save()
    if mode=='native':
        spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
        del model;gc.collect();d['live_model_tensors_after_release']=[n for n,r in refs if r() is not None];assert not d['live_model_tensors_after_release'];d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);save();assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2
    else:
        del llm;gc.collect()
    d['status']='complete';save();print('ACCURACY_WORKER_COMPLETE '+mode,flush=True)
if __name__=='__main__':main()
