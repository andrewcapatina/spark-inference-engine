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
original_argmax=torch.Tensor.argmax;reference=[];current=[];step=0;argmax_errors=[];logit_differences=[];generation_number=0
# The real GPU argmax is enqueued BEFORE diagnostic copies; added synchronization can still mask races.
def traced_argmax(t,*args,**kwargs):
 global step
 result=original_argmax(t,*args,**kwargs)
 if t.shape==(1,248320):
  host=t.cpu();gpu_index=int(result.item());cpu_index=int(original_argmax(host,-1).item());finite=bool(torch.isfinite(host).all());current.append(host)
  if gpu_index!=cpu_index or not finite:argmax_errors.append({'generation':generation_number,'step':step,'gpu_index':gpu_index,'cpu_index':cpu_index,'all_finite':finite});torch.save(host,out/f'argmax-error-{generation_number}-{step}.pt')
  if reference and not torch.equal(host,reference[step]):
   row={'generation':generation_number,'step':step,'different_elements':int((host!=reference[step]).sum()),'max_abs':float((host.float()-reference[step].float()).abs().nan_to_num().max()),'gpu_index':gpu_index,'cpu_index':cpu_index};logit_differences.append(row)
   if len(logit_differences)==1:torch.save({'reference':reference[step],'actual':host},out/'first-logit-difference.pt')
  step+=1
 return result
with torch.inference_mode():
 try:
  torch.Tensor.argmax=traced_argmax
  for generation_number in range(4):
   current=[];step=0;result=model.generate(ids,128,None);tokens=result[0,256:].tolist();row={'repetition':generation_number,'matches_canonical':tokens==expected,'steps':step,'argmax_errors':len(argmax_errors),'logit_differences':len(logit_differences),'output_token_ids':tokens};d['generations'].append(row);del result
   assert step==128
   if not reference:reference=current
   save();print(json.dumps({k:v for k,v in row.items() if k!='output_token_ids'}),flush=True)
   if tokens!=expected:break
 finally:torch.Tensor.argmax=original_argmax
 d['argmax_errors']=argmax_errors;d['logit_differences']=logit_differences;d['diagnostic_synchronization_caveat']='GPU argmax called before copies; extra CPU synchronization afterward can mask timing-dependent behavior';torch.save(torch.cat(reference),out/'reference-logits.pt');save()
del reference,current

del model,ids;gc.collect();assert not [n for n,t in refs if t() is not None]
spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;d['status']='complete';save();print('BASELINE_ONLY_DIAGNOSTIC_COMPLETE',flush=True)
