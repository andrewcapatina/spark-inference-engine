"""A/B retained local-Qwen methods on one loaded model; no implementation edits."""
import ast,hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import MethodType
import torch
sys.path.insert(0,'/workspace')
from spark_engine.cli import _load_qwen,_model_kind
run_name=sys.argv[1];root=Path('/workspace');out=root/'benchmark_runs'/run_name
source=(out/'qwen_model-before.py.txt').read_bytes()
manifest=json.loads((out/'paired-manifest.json').read_text())
assert hashlib.sha256(source).hexdigest()==manifest['engine_before']['spark_engine/qwen_model.py']
assert hashlib.sha256((root/'spark_engine/qwen_model.py').read_bytes()).hexdigest()==manifest['engine_candidate']['spark_engine/qwen_model.py']
tree=ast.parse(source.decode());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenCausalLM');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<retained-local-Qwen-baseline>','exec'),namespace)
original_generate=namespace['generate']
spec=importlib.util.spec_from_file_location('bench',root/'tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
torch.cuda.init();torch.cuda.set_per_process_memory_fraction(56*1024**3/torch.cuda.get_device_properties(0).total_memory)
print('LOADING_27B',flush=True)
text=_load_qwen(Path('/workspace/reference/qwen27b.gguf'),_model_kind(Path('/workspace/reference/qwen27b.gguf'),'auto'),'cuda');engine=text.engine
print('MODEL_LOADED',flush=True)
assert 'generate' not in vars(engine)
base=text.tokenizer.encode(bench.PROMPT,add_special_tokens=False);ids=torch.tensor([(base*((256+len(base)-1)//len(base)))[:256]],device='cuda')
def phase(label,**extra):
    import os
    with (out/'paired-phases.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),'phase':label,**extra})+'\n');f.flush();os.fsync(f.fileno())
result={'status':'running','protocol':'Paired diagnostic, same loaded27B weights,256 prompt, alternating AB/BA order; no profiler','source_manifest':manifest,'rows':[]}
path=out/'paired-results.json';bench.write_json(path,result)
with torch.inference_mode():
    for count in (8,128):
        for repetition in (-1,0,1,2,3):
            order=('before','after') if repetition%2==0 else ('after','before')
            for version in order:
                phase('before_generation',version=version,repetition=repetition,new_tokens=count)
                if version=='before':engine.generate=MethodType(original_generate,engine)
                try:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    generated=engine.generate(ids,max_new_tokens=count,eos_token_id=None)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
                    assert generated.shape[1]==256+count and torch.equal(generated[:,:256],ids)
                    row={'version':version,'repetition':repetition,'new_tokens':count,'wall_ms':elapsed,'output_sha256':bench.output_digest(generated,256),'memory':bench.memory(torch)}
                    phase('after_generation',**row)
                    del generated
                finally:
                    if version=='before':delattr(engine,'generate')
                if repetition>=0:result['rows'].append(row);bench.write_json(path,result);print(json.dumps({k:row[k] for k in ('version','new_tokens','repetition','wall_ms')}),flush=True)
        hashes={x['output_sha256'] for x in result['rows'] if x['new_tokens']==count};assert len(hashes)==1
assert 'generate' not in vars(engine)
import gc
del engine,text
gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
result['released_memory']=bench.memory(torch)
assert result['released_memory']['cuda_allocated_bytes']<=16*1024**2 and result['released_memory']['cuda_reserved_bytes']<=64*1024**2
result.update(status='complete',instance_method_restored=True);bench.write_json(path,result)
