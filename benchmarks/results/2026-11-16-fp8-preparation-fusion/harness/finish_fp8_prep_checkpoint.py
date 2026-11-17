"""Finalize documentation/status only after all acceptance and containment checks."""
from pathlib import Path
import json,subprocess,hashlib,ast,textwrap
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip());work=P/'benchmark_runs'/main.name
assert json.loads((main/'cleanup-check.json').read_text())['status']=='clean';assert json.loads((main/'protected-correctness.json').read_text())['status']=='all_protected_gates_passed';assert json.loads((main/'paired-prefill-audit.json').read_text())['status']=='fp8_prep_prefill_audit_passed'
engine=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();assert not subprocess.check_output(['git','diff','--','spark_engine'],cwd=P,text=True)
m=json.loads((main/'manifest.json').read_text());assert m['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
assert (P/'spark_engine/qwen_fp8_prepare.py').read_bytes()==(work/'prep_proto.py').read_bytes()
for number,filename,cls,fn in [('069','qwen_fp8.py','FP8Linear','forward'),('071','qwen_mlp_fp8_pair.py',None,'_make_pair_projector')]:
 tree=ast.parse((P/'spark_engine'/filename).read_text());container=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls) if cls else tree;node=next(n for n in container.body if isinstance(n,ast.FunctionDef) and n.name==fn);raw=ast.parse(textwrap.dedent((work/f'task-{number}-candidate.txt').read_text())).body[0];assert ast.dump(node)==ast.dump(raw)
(main/'final-source-ownership.json').write_text(json.dumps({'status':'local_qwen_ownership_verified','kernel':'068 verbatim','FP8Linear.forward':'069 AST exact, class indentation only','pair_factory':'071 verbatim','engine_commit':engine,'all_other17_original_engine_files_unchanged':True},indent=2)+'\n')
m.update(status='accepted_all_validations_complete',previous_accepted_engine_commit=m['accepted_engine_commit'],accepted_engine_commit=engine,retained_engine_commit=engine);(main/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
handoff=f'''# Engine development handoff

Current accepted engine: `{engine}` (fused FP8 activation preparation), on top
of e54057e shared-pair reuse and da80ee6 activation fusion. Local Qwen alone
wrote kernel068, FP8Linear.forward069, pair factory071; copied verbatim with
only class indentation. Codex supervised harness/prompts/references/placement.

New qwen_fp8_prepare.prepare_fp8 fuses scale/max/division/cast/pad for CUDA
contiguous BF16 inference rows>=16,width5120/6144. Other inputs and width17408
retain original preparation; decode unchanged. Original GEMMs/weights/scales,
BF16 output/fast accumulation unchanged. No persistent prepared-tensor cache.
Existing eight shared FP8 gate/up pairs still prepare once and use two GEMMs;
factory guards for hooks/overrides/subclasses/devices/shapes remain unchanged.
Reference CUDA scalar /448 follows rounded FP32 reciprocal multiply; tensor
x/scale follows RN division. Preserve clamp, round points, NaN bits, padding,
and disable contraction.9878/32640maxima differ if true division448 is used.

Incremental actual27B warm prefill:256tokens250.9062→241.7116ms save9.1946ms/
3.6646%;1024tokens985.0204→931.0772ms save53.9432ms/5.4764%. All8ABBAblocks
positive perlength; peak memory unchanged.16samples/backend/length,loading and
generation excluded. No output-token speed claim or additive percentages.
Profile217fused kernels replaces1519launches (-1302);233FP8/168NVFP4GEMMs
and GDNstate/key,activation,norm counts unchanged; prefill only,no decode.

Validated359 strict primitive cases under Torch2.13 AND originalTorch2.11,
including32640BF16finite maxima,boundaries,padding,Inf/NaN raw scale/byte bits;
54real-weight integration cases; full actual27B prefill+nextdecode logits/all
caches and8canonical128-ID replies exact;40answers/446NLL exact;all56Qwen/
15Llama protected gates incl original9B/27B numerical/conversation passed.
Report: benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md.
Main pointer: feedback/fp8-prep-main-run.txt. No active implementation task.

Rejected065unsafe padload,066NaN propagation,067unsupported Triton APIs,
070literal optional-import guard retained before local fixes. Previous rejected/
quarantined GDNstate-tail fusion,paired solves,independent chunk batching stay
rejected. Historical generation outlier did not recur; cause remains UNKNOWN.
Keep canonical fixtures and failing-logit capture; source-snapshot/commit
peelback plan remains, including uncommitted prototypes. Never weaken tests.

Previous history: e54057e FP8pair (shortflat,long3.125ms/.315%), da80ee6 MLP
activation (4.87ms/1.90%256;16.59ms/1.65%1024),f1e760f GDNkeys,3139a9b value
casts,ea70e47 gatednorm,dd139d0 norm,2beb3fd FP8max,6dd8d9f QKVreuse. Reports
under benchmarks/results preserve complete evidence. Do not add percentages.

Validation policy: diagnostic/tiny cases differ from real-checkpoint validation.
Require full real27B logits/caches/2x128IDs perbackend/length vs accepted oracle,
frozen40answers/446NLL,all56Qwen/15Llama gates. One large model at a time,
fresh processes for original9B and27B checkpoints. Native allocator32GiB,
primitives8GiB,original gates56GiB;host floor32GiB,new swap<=256MiB. All4
containers stopped exit0/OOMfalse,GPUidle,serviceMainPID0,sameboot,zero newSwap.
Keep STOP/GPU/LARGE holds and use guarded controllers. No packages/downloads/
subagents/unrelated work. Default engine authorship remains LOCAL QWEN ONLY;
earlier user exception was for GDN only. This context file is not training.
'''
assert len(handoff)<=12000;(P/'LOCAL_MODEL_HANDOFF.md').write_text(handoff)
p=P/'QWEN_TASK.md';s=p.read_text();a=s.index('Active work: fused FP8');b=s.index('Current user-authorized exception',a);s=s[:a]+f'''Completed fused FP8 preparation: accepted engine {engine}.
LocalQwen068kernel/069FP8forward/071pair placed verbatim; other17original
engine files unchanged.359strict cases pass in both runtimes,54realweight
integration cases exact; full27B outputs/caches/8canonical128IDs exact,
40answers/446NLL exact,all56Qwen/15Llama protected gates pass.
Warm prefill saves9.195ms/3.665%256 and53.943ms/5.476%1024,8/8positiveblocks
each,peakmemoryunchanged.217fused kernels remove1302launches; GEMMs/state math
unchanged. No additive-percentage or decode-speed claim. Allcontainersstopped,
GPUidle/serviceinactive,sameboot/zero newSwap,holdsretained. No active task.
Report benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md.
Historical outlier remains unexplained; raw rejected sources retained.

'''+s[b:];p.write_text(s)
(P/'PROGRESS.md').open('a').write(f'\n2026-11-17 UTC: Completed fused FP8 preparation, local-Qwen engine {engine}. `launch_fp8_prep_trial.py validation/cleanup`, `compare_fp8_prep_results.py`, `summarize_fp8_prep_protected.py` passed: full27B outputs/caches/8canonical128IDs,40answers/446NLL exact;359strict prep cases pass Torch2.13 AND originalTorch2.11;54real-weight cases;all56Qwen/15Llama incl original9B/27B numerical+conversation. Profile217fused prep kernels/1302launches removed;233FP8/168NVFP4GEMMs andGDN/norm/activation counts unchanged. Warm256250.9062→241.7116ms save9.1946ms/3.6646%;1024985.0204→931.0772ms save53.9432ms/5.4764%;8/8blocks positive each,peakmemoryunchanged. Local068/069/071 ownership verified;065/066/067/070 rejected retained. All4containers exit0/OOMfalse,GPUempty,service0,sameboot,zero newSwap,holdsretained. Historical outlier not observed,cause unknown. Report benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md. No active engine work; publication follows.\n')
report=P/'benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md';report.write_text(report.read_text().replace('ENGINE_COMMIT_PENDING',engine))
t=f'Completed fused FP8 preparation, local-Qwen engine {engine}. Warm prefill256250.906→241.712ms (-3.665%);1024985.020→931.077ms (-5.476%);8/8positiveblocks each,peakunchanged.1302fewer launches. Full27B outputs/caches/8canonical128IDs,40answers/446NLL exact;359strict cases both runtimes;all56Qwen/15Llama pass incl original9B/27B. Allcontainersstopped0/OOMfalse,GPUidle/service0,sameboot/zero newSwap,holdsretained. No active job. Historical outlier root unknown. Report benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md.\n';(R/'feedback/latest.txt').write_text(t);(P/'run_feedback/latest.txt').write_text(t);(R/'feedback/checkpoints.txt').open('a').write('\n'+t)
p=R/'status.json';s=json.loads(p.read_text());s.update(status='fp8_preparation_fusion_complete',phase='checkpointed',run_directory=str(main),active_run=str(main),current_fp8_prep_run=str(main),last_validated_commit=engine,experiment_idle=True,active_task=None,model='unsloth/Qwen3.8-27B-NVFP4',automatic_gpu_resume_held=True,remaining_failures=['Historical generation outlier cause unresolved; not reproduced in this work'],report=str(report));p.write_text(json.dumps(s,indent=2)+'\n');print(engine)
