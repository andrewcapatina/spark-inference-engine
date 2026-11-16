"""Read existing trace; attribute GDN kernels to their actual Torch CPU operator ancestors."""
from pathlib import Path
import sys
from collections import defaultdict
import json,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');run=Path((R/'feedback/prefill-value-main-run.txt').read_text().strip());stage=R/'feedback/nvfp4-reference'/sys.argv[1];result=json.loads((stage/'results.json').read_text());rows=[]
for case in result['cases']:
 path=Path(case['profile']['trace_summary']['trace']);path=stage/path.name;events=json.loads(path.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X'];ids={e.get('args',{}).get('correlation') for e in kernels};launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') in ids};spans=defaultdict(list);threads=defaultdict(list)
 for e in events:
  if e.get('cat') in ('cpu_op','user_annotation') and e.get('ph')=='X':spans[e['pid'],e['tid']].append(e)
 for key,e in launches.items():threads[e['pid'],e['tid']].append((e['ts'],key))
 owners={}
 for thread,items in threads.items():
  ordered=sorted(spans[thread],key=lambda e:(e['ts'],-e['dur']));i=0;stack=[]
  for ts,key in sorted(items):
   while i<len(ordered) and ordered[i]['ts']<=ts:
    e=ordered[i]
    while stack and stack[-1]['ts']+stack[-1]['dur']<=e['ts']:stack.pop()
    stack.append(e);i+=1
   while stack and stack[-1]['ts']+stack[-1]['dur']<=ts:stack.pop()
   owners[key]=stack.copy()
 top_ops={};leaf=defaultdict(lambda:{'kernels':0,'gpu_ms':0.});inclusive=defaultdict(lambda:{'kernels':0,'gpu_ms':0.});families=defaultdict(lambda:{'kernels':0,'gpu_ms':0.});total=0.;count=0
 for e in kernels:
  chain=owners.get(e.get('args',{}).get('correlation'),[]);names=[v['name'] for v in chain]
  if 'native.gdn.state_update' not in names:continue
  assert 'native.engine.prefill' in names
  duration=e['dur']/1000;count+=1;total+=duration;roots=[v for v in chain if v.get('cat')=='cpu_op'];root=roots[0] if roots else None
  if root is not None:
   key=(root['pid'],root['tid'],root['ts']);record=top_ops.setdefault(key,{'name':root['name'],'ts':root['ts'],'gdn_call_ts':next(v['ts'] for v in chain if v.get('name')=='native.gdn.state_update'),'gpu_ms':0.,'kernels':0});record['gpu_ms']+=duration;record['kernels']+=1
  ops=[v['name'] for v in chain if v.get('cat')=='cpu_op'];name=ops[-1] if ops else '<no-cpu-op>'
  for target,key in [(leaf,name),(families,e['name'])]:target[key]['kernels']+=1;target[key]['gpu_ms']+=duration
  for name in set(ops):inclusive[name]['kernels']+=1;inclusive[name]['gpu_ms']+=duration
 expected=next(x for x in case['profile']['trace_summary']['raw_kernel_ownership']['inclusive_scopes'] if x['name']=='native.gdn.state_update' and x['phase']=='prefill');assert count==expected['kernels'] and abs(total-expected['gpu_ms'])<1e-6
 def ordered(d):return [{'name':k,**v} for k,v in sorted(d.items(),key=lambda x:-x[1]['gpu_ms'])]
 sequence=sorted(top_ops.values(),key=lambda x:x['ts']);state_tail=[]
 for i in range(2,len(sequence)):
  triple=sequence[i-2:i+1]
  if [v['name'] for v in triple]==['aten::mul','aten::matmul','aten::add']:state_tail.append({'multiply':triple[0],'update_gemm':triple[1],'add':triple[2]})
 groups=defaultdict(list)
 for entry in sequence:groups[entry['gdn_call_ts']].append(entry)
 assert len(groups)==48
 cast_totals={'query':0.,'key':0.};cast_prefixes=[]
 for items in groups.values():
  assert [v['name'] for v in items[:2]]==['aten::to']*2
  cast_prefixes.append([{'name':v['name'],'gpu_ms':v['gpu_ms'],'kernels':v['kernels']} for v in items[:7]])
  for name,value in zip(cast_totals,items[:2]):cast_totals[name]+=value['gpu_ms']
 rhs_preparations=[]
 for j in range(4,len(sequence)):
  if sequence[j]['name']=='aten::linalg_solve_triangular' and [x['name'] for x in sequence[j-4:j]]==['aten::mul','aten::mul','aten::exp','aten::mul']:rhs_preparations.append(sequence[j-4:j])
 rhs_summary={'matched_preparations':len(rhs_preparations),'three_product_gpu_ms':sum(sum(x['gpu_ms'] for x in block if x['name']=='aten::mul') for block in rhs_preparations),'three_product_kernels':sum(sum(x['kernels'] for x in block if x['name']=='aten::mul') for block in rhs_preparations)}
 tail_summary={'matched_updates':len(state_tail),'multiply_gpu_ms':sum(x['multiply']['gpu_ms'] for x in state_tail),'add_gpu_ms':sum(x['add']['gpu_ms'] for x in state_tail),'matmul_gpu_ms':sum(x['update_gemm']['gpu_ms'] for x in state_tail),'multiply_kernels':sum(x['multiply']['kernels'] for x in state_tail),'add_kernels':sum(x['add']['kernels'] for x in state_tail)};assert len(state_tail)==48*case['prompt_tokens']//64
 rows.append({'initial_kernel_launching_operations_by_layer':cast_prefixes,'initial_fp32_cast_gpu_ms':cast_totals,'rhs_preparation':rhs_summary,'state_pointwise_tail':tail_summary,'prompt_tokens':case['prompt_tokens'],'gdn_gpu_ms':total,'gdn_kernels':count,'exclusive_leaf_operator_totals':ordered(leaf),'inclusive_operator_totals':ordered(inclusive),'kernel_families':ordered(families),'trace':str(path),'trace_bytes':path.stat().st_size,'baseline_engine_source_sha256':result['source']['sha256']})
 print(json.dumps({'prompt_tokens':case['prompt_tokens'],'initial_fp32_cast_gpu_ms':cast_totals,'rhs_preparation':rhs_summary,'state_pointwise_tail':tail_summary,'total_gpu_ms':total,'kernels':count,'leaf':ordered(leaf)[:12],'inclusive':ordered(inclusive)[:12]},indent=2),flush=True)
d={'status':'cpu_diagnostic_complete','scope':'Existing real27B one-token/zero-decode profile only; no GPU run or engine edits. Inclusive totals overlap; exclusive leaf totals partition GDN kernels.','cases':rows};(run/('gdn-cast-breakdown-'+stage.name+'.json')).write_text(json.dumps(d,indent=2)+'\n')
