"""Real downloaded NVFP4 native smoke/identity/conversation; not a benchmark."""
from pathlib import Path
import sys,json,time,gc,hashlib,importlib.util

def main():
 import torch
 sys.path.insert(0,'/workspace')
 from spark_engine.qwen_text import NativeQwenText
 from spark_engine.qwen_nvfp4 import NVFP4Linear
 from spark_engine.qwen_fp8 import FP8Linear
 spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 out=Path('/results')/sys.argv[1];out.mkdir(exist_ok=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
 d={'status':'loading','scope':'Actual Unsloth27B native quantized smoke, exact8token reference comparison and2turn arithmetic conversation','cases':[],'conversation':[],'before_load_memory':bench.memory(torch)}
 def save():bench.write_json(out/'results.json',d)
 def phase(label,**extra):
  with (out/'phases.jsonl').open('a') as f:
   import os
   f.write(json.dumps({'time':time.time(),'phase':label,**extra})+'\n');f.flush();os.fsync(f.fileno())
 save();phase('before_load');start=time.perf_counter();text=NativeQwenText.from_pretrained('/model');engine=text.engine;d['load_ms']=(time.perf_counter()-start)*1000;d['loaded_memory']=bench.memory(torch);d['metadata']=engine.quantization_metadata
 assert sum(isinstance(m,NVFP4Linear) for m in engine.modules())==168
 assert sum(isinstance(m,FP8Linear) for m in engine.modules())==233
 assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in engine.parameters()) and all(b.device.type=='cuda' for b in engine.buffers())
 def resident(model):
  seen=set();size=0
  for t in list(model.parameters())+list(model.buffers()):
   storage=t.untyped_storage();key=storage.data_ptr()
   if key not in seen:seen.add(key);size+=storage.nbytes()
  return size
 d['resident_tensor_bytes']=resident(engine);assert d['resident_tensor_bytes']<28*1024**3;d['status']='generating';save();phase('after_load',load_ms=d['load_ms']);print('NATIVE_REAL_NVFP4_LOADED '+json.dumps(d['metadata']),flush=True)
 reference=json.loads(Path('/results/results.json').read_text())
 with torch.inference_mode():
  for case in reference['cases']:
   ids=torch.tensor([case['prompt_token_ids']],device='cuda');length=ids.shape[1];phase('before_generation',prompt_tokens=length);torch.cuda.synchronize();start=time.perf_counter();generated=engine.generate(ids,8,None);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000;tokens=generated[0,length:].tolist();expected=case['samples'][0]['output_token_ids'][:8]
   row={'prompt_tokens':length,'output_tokens':len(tokens),'output_token_ids':tokens,'vllm_reference_first8':expected,'reference_exact_match':tokens==expected,'wall_ms_diagnostic_only':elapsed,'memory':bench.memory(torch)};d['cases'].append(row);save();phase('after_generation',**{k:v for k,v in row.items() if k!='memory'});assert tokens==expected,'Inspect exact reference mismatch; vLLM usesFP8KV while nativeBF16KV, needs matching-precision diagnostic'
  messages=[{'role':'system','content':'Answer with a single digit and nothing else.'},{'role':'user','content':'What is 2 + 2?'}]
  first=text.chat(messages,max_new_tokens=16);d['conversation'].append({'question':messages[-1]['content'],'answer':first});save();assert first.strip()=='4',repr(first)
  messages.extend([{'role':'assistant','content':first},{'role':'user','content':'Add 3 to that number. What is the result?'}]);second=text.chat(messages,max_new_tokens=16);d['conversation'].append({'question':messages[-1]['content'],'answer':second});save();assert second.strip()=='7',repr(second)
 d['status']='complete';d['final_memory']=bench.memory(torch);del ids,generated,engine,text;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize();d['released_memory']=bench.memory(torch);save();phase('released');print('NATIVE_REAL_NVFP4_COMPLETE',flush=True)
if __name__=='__main__':main()
