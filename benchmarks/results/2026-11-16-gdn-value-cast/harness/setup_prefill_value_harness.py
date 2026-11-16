"""Adapt existing unchanged strict test coverage and isolation to a cast-only candidate."""
from pathlib import Path
import ast
R=Path('/home/cap/spark-inference-overnight-20261113')
s=(R/'acceptance/check_gdn_prepared_integration.py').read_text().replace(' import spark_engine.qwen_gdn_prepared_solve as tail\n','').replace('qwen_ops_before_solve_processing','qwen_ops_before_value_processing')
a=s.index('  tail_before=');b=s.index('  previous={name:',a)
s=s[:a]+'''  float_before=torch.Tensor.float;value_casts=[0]
  def observe_float(t,*a,**kw):
   if t.data_ptr()==args[2].data_ptr():value_casts[0]+=1
   return float_before(t,*a,**kw)
  torch.Tensor.float=observe_float
'''+s[b:]
s=s.replace('   tail.solve_prepared=tail_before','   torch.Tensor.float=float_before')
a=s.index('  assert tail_calls[');b=s.index('  return result,counts[0]',a)
s=s[:a]+'''  eligible=fn is candidate and args[0].dtype==torch.bfloat16 and args[2].dtype==torch.bfloat16 and args[2].is_cuda and torch.is_inference_mode_enabled()
  assert value_casts[0]==(0 if eligible else 1),('Unexpected value float calls',value_casts,eligible)
  assert result[1].dtype==torch.float32 and result[0].dtype==args[0].dtype
'''+s[b:]
a=s.index('   args=inputs(1,65,2,16,16,torch.bfloat16);previous_import=');b=s.index('   args=inputs(1,65,2,16,16,torch.bfloat16);state=',a)
s=s[:a]+s[b:];s=s.replace("   d['missing_triton_only_fallback_and_helper_errors_propagate']=True;",'   ')
s=s.replace('GDN_PREPARED_INTEGRATION','GDN_VALUE_INTEGRATION')
# Extend existing strict comparisons to CPU and non-FP32 provided state.
needle='  with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:'
extra='''  with torch.inference_mode():
   for dtype in [torch.float32,torch.bfloat16,torch.float64]:
    args=tuple(x.cpu().to(dtype) for x in inputs(1,65,2,16,32,torch.float32))
    for state_dtype in [None,torch.bfloat16,torch.float64]:
     state=None if state_dtype is None else torch.randn(1,2,16,32,dtype=state_dtype)
     a,_=counted(original,args,state,64);b,_=counted(candidate,args,state,64);equal(a,b)
   for state_dtype in [torch.bfloat16,torch.float64]:
    args=inputs(1,65,2,16,32,torch.bfloat16);state=torch.randn(1,2,32,16,device='cuda',dtype=state_dtype).transpose(-1,-2);before=state.clone()
    a,_=counted(original,args,state,64);b,_=counted(candidate,args,state,64);equal(a,b);equal(state,before)
   del args,state,a,b,before
   d['cpu_and_provided_state_dtype_cases']=11
'''
assert needle in s;s=s.replace(needle,extra+needle)
(R/'acceptance/check_gdn_value_integration.py').write_text(s)
s=(R/'acceptance/benchmark_prefill_solve_ab.py').read_text().replace('qwen_ops_before_solve_processing','qwen_ops_before_value_processing')
s=s.replace("import_start=time.perf_counter();import spark_engine.qwen_gdn_prepared_solve as tail;tail_import_seconds=time.perf_counter()-import_start;",'').replace("'tail_module_import_seconds':tail_import_seconds,",'')
a=s.index(' def counted_call(');b=s.index(' save();start=time.perf_counter();model=',a)
s=s[:a]+s[b:]
s=s.replace("'tail_calls':{},",'')
s=s.replace("select(backend);(logits,cache),count,elapsed=counted_call(model.prefill,ids);row['tail_calls'][backend]=count;row['first_validation_prefill_wall_seconds'][backend]=elapsed;assert count==(0 if backend=='original' else 48*((length+63)//64))","select(backend);torch.cuda.synchronize();started=time.perf_counter();logits,cache=model.prefill(ids);torch.cuda.synchronize();row['first_validation_prefill_wall_seconds'][backend]=time.perf_counter()-started")
s=s.replace('(step_logits,step_cache),decode_count,_=counted_call(model.decode,next_id,cache);assert decode_count==0','step_logits,step_cache=model.decode(next_id,cache)')
s=s.replace("['original','candidate','candidate','original']*2", "['original','candidate','candidate','original']*4")
s=s.replace('PREFILL_SOLVE_AB','PREFILL_VALUE_AB')
(R/'acceptance/benchmark_prefill_value_ab.py').write_text(s)
for kind in ['integration','ab']:
 s=(R/('run_prefill_solve_'+kind+'.py')).read_text().replace('check_gdn_prepared_integration.py','check_gdn_value_integration.py').replace('benchmark_prefill_solve_ab.py','benchmark_prefill_value_ab.py').replace('GDN prepared RHS plus paired triangular solve','GDN native Torch value cast deferral').replace('CPU/grad/import fallbacks','CPU/grad fallback paths')
 (R/('run_prefill_value_'+kind+'.py')).write_text(s)
for filename in ['prepare_prefill_solve_trial.py','launch_prefill_solve_prepared.py']:
 s=(R/filename).read_text().replace('prefill-solve','prefill-value').replace('prefill_solve','prefill_value').replace('User authorized GDN RHS preparation/paired triangular solve; preserve original FP32 products, eager exps and state/GEMM arithmetic','User authorized GDN value cast deferral; preserve FP32 state/output, original solves and eager arithmetic')
 (R/filename.replace('prefill_solve','prefill_value')).write_text(s)
for path in [R/'acceptance/check_gdn_value_integration.py',R/'acceptance/benchmark_prefill_value_ab.py',R/'run_prefill_value_integration.py',R/'run_prefill_value_ab.py']:
 ast.parse(path.read_text());assert 'import spark_engine.qwen_gdn_prepared_solve' not in path.read_text()
print('VALUE_HARNESS_READY: strict298direct/32blocks/20tiny/2gradient plus11CPU/state cases;16ABBAprefills and2x128IDs/backend/length')
