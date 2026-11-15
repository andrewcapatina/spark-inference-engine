"""CPU profiling summaries; all CUDA kernels count, including non-aten custom ops."""
from pathlib import Path
from collections import defaultdict
import json,gzip,re

def interval_union(intervals):
    end=None;total=0.
    for start,finish in sorted(intervals):
        if end is None or start>end:total+=finish-start;end=finish
        elif finish>end:total+=finish-end;end=finish
    return total

def trace_summary(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:data=json.load(f)
    memory=[e.get('args',{}) for e in data.get('traceEvents',[]) if e.get('name')=='[memory]' and e.get('args',{}).get('Device Type')==1]
    memory_summary={'allocation_events':sum(x.get('Bytes',0)>0 for x in memory),'free_events':sum(x.get('Bytes',0)<0 for x in memory),'allocated_bytes_sum':sum(max(0,x.get('Bytes',0)) for x in memory),'freed_bytes_sum':sum(max(0,-x.get('Bytes',0)) for x in memory),'peak_total_allocated_bytes':max((x.get('Total Allocated',0) for x in memory),default=0),'peak_total_reserved_bytes':max((x.get('Total Reserved',0) for x in memory),default=0),'note':'Allocator request totals are cumulative, not live memory or measured memory bandwidth.'}
    events=data.get('traceEvents',[]);kernels=defaultdict(lambda:{'calls':0,'gpu_ms':0.});intervals=[];copies=0.;count=0
    for e in events:
        cat=e.get('cat','');duration=e.get('dur',0)
        if cat not in ('kernel','gpu_memcpy','gpu_memset') or e.get('ph')!='X':continue
        intervals.append((e['ts'],e['ts']+duration))
        if cat=='kernel':
            row=kernels[e['name']];row['calls']+=1;row['gpu_ms']+=duration/1000;count+=1
        else:copies+=duration/1000
    assert count>0,'Trace has no CUDA kernels; not a GPU profile'
    span=max(b for a,b in intervals)-min(a for a,b in intervals);active=interval_union(intervals)
    return {'raw_kernel_ownership':raw_kernel_ownership(events),'allocator_events':memory_summary,'trace':str(path),'trace_bytes':path.stat().st_size,'cuda_kernel_count':count,'kernel_gpu_ms_sum':sum(r['gpu_ms'] for r in kernels.values()),'copy_memset_gpu_ms_sum':copies,'gpu_activity_span_ms':span/1000,'gpu_active_union_ms':active/1000,'gpu_gap_fraction_within_activity_span':1-active/span if span else 0,'kernel_time_note':'Sums may overlap across streams; active union avoids double counting. Gaps are profiled, not unprofiled launch-overhead measurements.','kernels':[{'name':name,**row} for name,row in sorted(kernels.items(),key=lambda kv:kv[1]['gpu_ms'],reverse=True)[:40]]}

def phase_and_category(event):
    names=[];node=event
    while node is not None:
        names.append(node.name);node=node.cpu_parent
    phase='prefill' if 'native.engine.prefill' in names else 'decode' if 'native.engine.decode' in names else 'generation_other'
    if phase=='generation_other':
        for name in names:
            match=re.fullmatch(r'execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)',name)
            if match:
                context,decode=map(int,match.groups());phase='prefill' if context and not decode else 'decode' if decode and not context else 'mixed' if context and decode else 'generation_other'
                break
    priorities=[('primitive.nvfp4.gemm','nvfp4_gemm'),('primitive.nvfp4.quantize','fp4_quantization'),('primitive.fp8.gemm','fp8_gemm'),('native.quant.fp8','fp8_activation_padding_misc'),('native.quant.nvfp4','nvfp4_scale_misc'),('native.gdn.state_update','gdn_state_math'),('native.layer.norm','normalization'),('native.layer.attention','attention_other'),('native.layer.gdn','gdn_other'),('native.layer.mlp','mlp_activation_other')]
    return phase,next((category for scope,category in priorities if scope in names),'model_other')

def event_summary(prof):
    scopes={};ops={};categories={}
    for e in prof.events():
        if not str(e.device_type).endswith('CPU') or e.name.startswith('Activity Buffer '):continue
        phase,category=phase_and_category(e);key=(phase,category);row=categories.setdefault(key,{'calls':0,'self_cpu_ms':0.,'self_gpu_ms':0.,'allocated_bytes_sum':0,'freed_bytes_sum':0})
        row['calls']+=1;row['self_cpu_ms']+=e.self_cpu_time_total/1000;row['self_gpu_ms']+=e.self_device_time_total/1000;row['allocated_bytes_sum']+=max(0,e.self_device_memory_usage);row['freed_bytes_sum']+=max(0,-e.self_device_memory_usage)
        if e.name.startswith(('native.','primitive.')):
            row=scopes.setdefault((phase,e.name),{'calls':0,'total_cpu_ms':0.,'self_cpu_ms':0.,'total_gpu_ms':0.})
            row['calls']+=1;row['total_cpu_ms']+=e.cpu_time_total/1000;row['self_cpu_ms']+=e.self_cpu_time_total/1000;row['total_gpu_ms']+=e.device_time_total/1000
        else:
            row=ops.setdefault((phase,e.name),{'calls':0,'self_cpu_ms':0.,'self_gpu_ms':0.,'allocated_bytes_sum':0,'freed_bytes_sum':0})
            row['calls']+=1;row['self_cpu_ms']+=e.self_cpu_time_total/1000;row['self_gpu_ms']+=e.self_device_time_total/1000;row['allocated_bytes_sum']+=max(0,e.self_device_memory_usage);row['freed_bytes_sum']+=max(0,-e.self_device_memory_usage)
    return {'category_self_times':[{'phase':p,'category':c,**r} for (p,c),r in categories.items()],'inclusive_scopes':[{'phase':p,'name':n,**r} for (p,n),r in scopes.items()],'operators_by_self_gpu':[{'phase':p,'name':n,**r} for (p,n),r in sorted(ops.items(),key=lambda kv:kv[1]['self_gpu_ms'],reverse=True)[:80]],'operators_by_self_cpu':[{'phase':p,'name':n,**r} for (p,n),r in sorted(ops.items(),key=lambda kv:kv[1]['self_cpu_ms'],reverse=True)[:50]],'operators_by_allocations':[{'phase':p,'name':n,**r} for (p,n),r in sorted(ops.items(),key=lambda kv:kv[1]['allocated_bytes_sum'],reverse=True)[:40]],'notes':'CUPTI Activity Buffer bookkeeping is excluded because it duplicates GPU attribution. Scope totals are inclusive and overlap; categories use exclusive self times. Allocation sums are cumulative traffic, not peak live memory.'}

def raw_kernel_ownership(events):
    """Join GPU correlation IDs to CPU launch timestamps and containing scopes.
    FlashInfer kernels can lack Torch External IDs; CUDA correlation remains available.
    """
    gpu=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    ids={e.get('args',{}).get('correlation') for e in gpu};ids.discard(None)
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_driver','cuda_runtime') and e.get('args',{}).get('correlation') in ids}
    annotations=defaultdict(list);by_thread=defaultdict(list)
    for e in events:
        if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name','').startswith(('native.','primitive.','execute_context_')):annotations[(e['pid'],e['tid'])].append(e)
    for correlation,e in launches.items():by_thread[(e['pid'],e['tid'])].append((e['ts'],correlation))
    owners={}
    for thread,rows in by_thread.items():
        scopes=sorted(annotations[thread],key=lambda e:(e['ts'],-e['dur']));index=0;stack=[]
        for timestamp,correlation in sorted(rows):
            while index<len(scopes) and scopes[index]['ts']<=timestamp:
                scope=scopes[index]
                while stack and stack[-1]['ts']+stack[-1]['dur']<=scope['ts']:stack.pop()
                stack.append(scope);index+=1
            while stack and stack[-1]['ts']+stack[-1]['dur']<=timestamp:stack.pop()
            owners[correlation]=[e['name'] for e in stack]
    categories={};scope_totals={};category_kernels={};matched=0.;unmatched=0.;count=0
    from types import SimpleNamespace
    for kernel in gpu:
        correlation=kernel.get('args',{}).get('correlation');names=owners.get(correlation,[]);duration=kernel['dur']/1000
        node=None
        for name in names:node=SimpleNamespace(name=name,cpu_parent=node)
        phase,category=phase_and_category(node) if node is not None else ('generation_other','model_other')
        if not any(n.startswith(('native.','primitive.')) for n in names):
            label=kernel['name'].lower()
            if 'blockscaled' in label and 'gemm' in label:category='nvfp4_gemm'
            elif 'gemm' in label and 'e4m3' in label:category='fp8_gemm'
            elif 'gemm' in label:category='other_gemm'
            elif any(x in label for x in ('recurrent_gated_delta','flashinfergdn','gated_delta_rule')):category='gdn_fused_state'
            elif 'norm' in label:category='normalization_or_fused_quant'
            elif 'quant' in label or 'cvt_fp16_to_fp4' in label:category='quantization'
            elif 'flash_fwd' in label or 'attn' in label or 'attention' in label:category='attention'
            else:category='other_or_fused_gpu'

        row=categories.setdefault((phase,category),{'kernels':0,'gpu_ms':0.});row['kernels']+=1;row['gpu_ms']+=duration
        detail=category_kernels.setdefault((phase,category,kernel['name']),{'calls':0,'gpu_ms':0.});detail['calls']+=1;detail['gpu_ms']+=duration
        for name in set(names):
            row=scope_totals.setdefault((phase,name),{'kernels':0,'gpu_ms':0.});row['kernels']+=1;row['gpu_ms']+=duration
        if correlation in launches:matched+=duration;count+=1
        else:unmatched+=duration
    details={}
    for (phase,category,name),row in sorted(category_kernels.items(),key=lambda kv:kv[1]['gpu_ms'],reverse=True):
        key=phase+':'+category
        if len(details.setdefault(key,[]))<5:details[key].append({'name':name,**row})
    return {'largest_kernels_by_phase_category':details,'categories':[{'phase':p,'category':c,**r} for (p,c),r in categories.items()],'inclusive_scopes':[{'phase':p,'name':n,**r} for (p,n),r in scope_totals.items()],'launch_matched_kernels':count,'launch_unmatched_kernels':len(gpu)-count,'launch_matched_gpu_ms':matched,'launch_unmatched_gpu_ms':unmatched,'method':'CUDA kernel correlation -> CPU launch timestamp/thread -> active native record scopes or explicit vLLM context/generation labels. Reference component families use kernel names; fused work is not split. Raw kernels count once; scope totals overlap.','native_annotation_events':sum(map(len,annotations.values()))}
