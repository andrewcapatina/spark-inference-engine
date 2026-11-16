"""Record rejected experiment without claiming diagnostic success as implementation validation."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-state-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());assert m['status']=='quarantined_and_exact_baseline_restored';assert m['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert json.loads((run/'cleanup-check.json').read_text())['status']=='clean';out=P/'benchmarks/results/2026-11-16-gdn-state-tail';out.mkdir(parents=True,exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);B=R/'feedback/nvfp4-reference'
for kind in ['small','integration','ab','diagnostic']:
 stage=Path((R/f'feedback/prefill-state-{kind}-run.txt').read_text().strip());shutil.copy2(B/stage.name/'results.json',out/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',out/(kind+'-trial-status.json'))
 if kind=='ab':shutil.copy2(stage/'27b-worker.log',out/'failed-real-ab.log')
for name in ['manifest.json','gdn-operator-breakdown.json','cleanup-check.json','small-controller-before-label-correction.py.txt','small-trial-status-before-label-correction.json']:shutil.copy2(run/name,out/name)
for p in (P/'benchmark_runs'/run.name).iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
files=['diagnose_gdn_profile_cpu.py','quarantine_state_tail.py','check_state_quarantine_cleanup.py','report_state_quarantine.py','request_prefill_state_author.py','prepare_prefill_state_trial.py','launch_prefill_state_prepared.py','place_prefill_state_helper.py','place_prefill_state_integration.py','run_prefill_state_small.py','run_prefill_state_integration.py','run_prefill_state_ab.py','run_prefill_state_diagnostic.py','nvfp4_update_runtime.py','shutdown_process_probe.py','acceptance/check_gdn_state_tail.py','acceptance/check_gdn_state_integration.py','acceptance/benchmark_prefill_state_ab.py','acceptance/diagnose_state_generation.py','acceptance/qwen_ops_before_state_processing.py.txt','acceptance/request_nvfp4_candidate.py']+[f'prompts/nvfp4-task-{n:03d}.txt' for n in [41,42,43,44]]
for name in files:shutil.copy2(R/name,h/Path(name).name)
(h/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir() if p.name!='SHA256.json'},indent=2)+'\n')
(out/'README.md').write_text('''# GDN state pointwise tail — quarantined, not adopted

The accepted 16-file engine remains **ea70e47** (reportcf6b6a6), restored byte for byte after retaining the complete17-file experimental engine and patch. No speedup from this package is accepted. User code was authored solely by local Qwen; the supervisor supplied diagnostics, tests, prompts, scheduling and exact rollback.

## Why this target

Existing real27B 1,024-token, zero-decode profiling attributes357.82ms/30,960kernels to GDN. Exclusive leaf costs include multiplication98.32ms, matrix multiplication69.00ms, copies64.93ms and solve kernels54.81ms. Solve inclusive time63.99ms includes internal work and overlaps leaf costs; totals must not be added together. Correlation/timestamp attribution confirms the final per-chunk state multiply/add pair alone costs25.55ms/1,536kernels over768updates. Fusing only that pair avoids the arithmetic changes that broke the older whole-GDN compiler candidate.

Local043 implements a fresh FP32 output for state*decay + original eager update-GEMM, with floating-point fusion disabled. Local044 adds optional BF16 CUDA inference dispatch only inside gated_delta_chunk; all exponentials, statistics, solves and GEMMs remain original Torch. Cached recurrent decode is unchanged. Local041 was rejected before placement for missing constexpr row size;042 for handling a missing import name incorrectly. All raw responses/reviews are retained.

## Actual results and failure

- Standalone136 bitwise/ownership/fallback cases pass, including signedzero/subnormal/extreme-finite values and an explicit FMA counterexample; two gradient fallbacks and every transition in1/16/128/768-step state chains match exactly. Isolated real-shape tail13.93→10.95microseconds is not a checkpoint speedup.
- Integrated298 direct GDN cases,32dense/mixed-quantized block cases,20tiny whole-model cases and two gradient checks pass. Coverage includes arbitrary chunk boundaries, carried state, noncontiguous initial state, unchanged masks/continuation, CPU/no_grad/autograd fallback and import/helper error propagation. Tiny prefill/forward/nextdecode/cache/16IDs match exactly.
- Real27B at256tokens initially matches entire prefill logits/all cache tensors and the complete next decode. The subsequent128-token greedy comparison FAILS at zero-based index36 (the37th output token). The1,024-token case is not reached. Partial timings from this failed run are not accepted performance evidence.
- A fresh diagnostic repeats original generation twice and candidate generation twice: all four128-token sequences match the failed run's original baseline. Original-vs-original and original-vs-candidate then preserve all logits/every cache tensor and cache layouts throughout127teacher-forced decode steps. The original failure is not reproduced or explained by these successful diagnostics; it is retained as a failed gate, never counted as passing.

No numerical tolerances/assertions were weakened. No speculative engine repair was made. The unexplained generation failure prevents adoption, consistent with the earlier convolution-history quarantine. Candidate quality/profile/protected suites were not pursued after the failed gate. The restored accepted source already has the separately documented56Qwen/15Llama and quality validations; this is not a claim that the rejected candidate passed those suites.

## Containment and provenance

The full candidate was retained BEFORE rollback under the ignored local benchmark artifacts. Changed before/after source, raw responses, review decisions, rejected patch, source hashes, failed log, result JSON and exact harness are included here. The small trial initially inherited a misleading normalization scope label from its controller template; original metadata/controller and a clearly labelled corrected status are retained. Numerical artifacts and engine source were unchanged by that label correction.

All four experiment containers are stopped with exit0/OOMfalse, no GPU compute processes, serviceMainPID0, same boot, zero new per-worker swap, STOP and GPU holds retained. Large models ran serially. See cleanup-check.json for actual telemetry and image IDs.

Model: unsloth/Qwen3.8-27B-NVFP4, revision f0b7c9e722f5565102fff8481c99e4d86ae099c7. Installed pinned Torch2.13+cu130 runtime. Main run: '''+str(run)+'''. Raw weights, compiler caches and large traces remain outside Git.

The next independent experiment investigates pairing the two triangular right-hand-side solves, first requiring exact outputs and worthwhile isolated performance before integration. The state-tail candidate remains excluded.
''');print(out/'README.md')
