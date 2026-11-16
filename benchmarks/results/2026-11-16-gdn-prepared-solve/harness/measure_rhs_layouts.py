"""Measure actual-convolution-style RHS strides, same tensors per method; no checkpoint loaded."""
from pathlib import Path
import sys,json,hashlib,gc,statistics,importlib.util,importlib.machinery
sys.path.insert(0,'/workspace')
def main():
 import torch
 import spark_engine.qwen_gdn_rhs_pack as current
 run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(52052);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 loader=importlib.machinery.SourceFileLoader('retained_flat_rhs','/harness/flat_rhs047.py.txt');spec=importlib.util.spec_from_loader(loader.name,loader);flat=importlib.util.module_from_spec(spec);sys.modules[loader.name]=flat;loader.exec_module(flat)
 d={'status':'running','mode':mode,'scope':'Layout-specific isolated RHS packing only, not model validation or whole-engine speedup','cases':[],'flat_source_sha256':hashlib.sha256(Path('/harness/flat_rhs047.py.txt').read_bytes()).hexdigest(),'candidate_source_sha256':hashlib.sha256(Path('/workspace/spark_engine/qwen_gdn_rhs_pack.py').read_bytes()).hexdigest()}
 def save():(out/'results.json').write_text(json.dumps(d,indent=2)+'\n')
 def reference(v,k,b,e):return torch.cat((v*b.unsqueeze(-1),(k*b.unsqueeze(-1))*e.unsqueeze(-1)),dim=-1)
 methods={'original':reference,'flat':flat.pack_rhs}
 if d['flat_source_sha256']!=d['candidate_source_sha256']:methods['candidate']=current.pack_rhs
 save()
 with torch.inference_mode():
  for length in [256,1024]:
   for layout in ['feature_contiguous','conv_time_contiguous']:
    if layout=='conv_time_contiguous':v=torch.randn(1,48,128,length,device='cuda').transpose(-1,-2)[:,:,:64,:]
    else:v=torch.randn(1,length,48,128,device='cuda').transpose(1,2)[:,:,:64,:]
    k=torch.randn(1,length,48,128,device='cuda').transpose(1,2)[:,:,:64,:];beta=torch.rand(1,length,48,device='cuda').transpose(1,2)[:,:,:64];exp=torch.rand(1,48,64,device='cuda');args=(v,k,beta,exp);expected=reference(*args);row={'length_source':length,'layout':layout,'input_strides':[list(x.stride()) for x in args],'samples':[]}
    for name,fn in methods.items():
     actual=fn(*args);assert torch.equal(actual.view(torch.int32),expected.view(torch.int32));del actual
     for _ in range(3):actual=fn(*args);del actual
    order=['original','flat','flat','original']*2 if len(methods)==2 else ['original','flat','candidate','candidate','flat','original']*2
    for name in order:
     fn=methods[name];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(32):value=fn(*args)
     b.record();b.synchronize();row['samples'].append({'backend':name,'cuda_ms_per_pack':a.elapsed_time(b)/32});del value
    row['medians']={name:statistics.median(x['cuda_ms_per_pack'] for x in row['samples'] if x['backend']==name) for name in methods};d['cases'].append(row);save();print(json.dumps({'layout':layout,'length_source':length,'medians':row['medians']}),flush=True);del v,k,beta,exp,args,expected,a,b
 gc.collect();spec=importlib.util.spec_from_file_location('bench','/workspace/tools/benchmark_qwen.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench);d['released_library_workspaces']=bench.clear_cuda_library_workspaces(torch);d['released_memory']=bench.memory(torch);assert d['released_memory']['cuda_allocated_bytes']<=16*1024**2 and d['released_memory']['cuda_reserved_bytes']<=64*1024**2;d['status']='complete';save()
if __name__=='__main__':main()
