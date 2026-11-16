"""Clean baseline, no candidate import: repeatability and allocator-order probes."""
from pathlib import Path
import sys,json,os,time,gc,hashlib,weakref,importlib.util,inspect
sys.path.insert(0,'/workspace')
import torch
from transformers import PreTrainedTokenizerFast
from spark_engine.qwen_model import QwenCausalLM
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
assert not Path('/workspace/spark_engine/qwen_key_prepare.py').exists()
canonical=json.loads(Path('/harness/prefill-canonical-generation-v1.json').read_text());case=next(x for x in canonical['cases'] if x['prompt_tokens']==256)
d={'status':'loading','mode':mode,'scope':'Unmodified baseline only; candidate absent and never imported; fixed accepted prompt+128IDs','generations':[],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/workspace/spark_engine').glob('*.py')}}
def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
save();model=QwenCausalLM.from_pretrained('/model');refs=[(n,weakref.ref(t)) for n,t in list(model.named_parameters())+list(model.named_buffers())];ids=torch.tensor([case['prompt_token_ids']],device='cuda');expected=case['output_token_ids'];d['status']='diagnosing';save();print('BASELINE_ONLY_MODEL_LOADED',flush=True)
import flashinfer
d['flashinfer_functions']={name:{'signature':str(inspect.signature(getattr(flashinfer,name))),'source':inspect.getsource(getattr(flashinfer,name))} for name in ['fp4_quantize','mm_fp4']};save()
with torch.inference_mode():
 for phase in ['cold','prefill_primed','allocator_perturbed']:
  if phase=='prefill_primed':
   for _ in range(20):result=model.prefill(ids);del result
  for repetition in range(6):
   if phase=='allocator_perturbed':
    junk=[torch.full((n,),.125+repetition,device='cuda',dtype=torch.float32) for n in [512,393216,786432,1048576]];del junk
   start=time.perf_counter();result=model.generate(ids,128,None);tokens=result[0,256:].tolist();record={'phase':phase,'repetition':repetition,'elapsed_seconds':time.perf_counter()-start,'matches_canonical':tokens==expected,'first_mismatch':next((i for i,(x,y) in enumerate(zip(tokens,expected)) if x!=y),None),'output_token_ids':tokens};d['generations'].append(record);del result;save();print(json.dumps({k:v for k,v in record.items() if k!='output_token_ids'}),flush=True)
   if tokens!=expected:
    d['first_failure']={k:v for k,v in record.items()};break
  if 'first_failure' in d:break
del model,ids;gc.collect();assert not [n for n,t in refs if t() is not None]
spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;d['status']='complete';save();print('BASELINE_ONLY_DIAGNOSTIC_COMPLETE',flush=True)
