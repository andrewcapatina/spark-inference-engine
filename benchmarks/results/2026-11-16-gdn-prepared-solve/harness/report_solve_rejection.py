"""Archive numerically correct but unhelpful GDN solver experiments after exact rollback."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-solve-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());assert m['status']=='not_adopted_no_meaningful_prefill_gain_baseline_restored';assert m['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert json.loads((run/'cleanup-check.json').read_text())['status']=='clean';out=P/'benchmarks/results/2026-11-16-gdn-prepared-solve';out.mkdir(parents=True,exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);B=R/'feedback/nvfp4-reference'
for kind in ['small','pack','integration','ab','profile','layout','variants']:
 stage=Path((R/f'feedback/prefill-solve-{kind}-run.txt').read_text().strip());shutil.copy2(B/stage.name/'results.json',out/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',out/(kind+'-trial-status.json'))
for path in run.glob('*.json'):shutil.copy2(path,out/path.name)
for path in run.glob('*.py.txt'):shutil.copy2(path,out/path.name)
for p in (P/'benchmark_runs'/run.name).iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
files=['request_prefill_solve_author.py','request_prefill_solve_integration_tasks.py','prepare_prefill_solve_trial.py','launch_prefill_solve_prepared.py','place_prefill_solve_helper.py','place_prefill_solve_pack.py','place_prefill_solve_prepared_module.py','place_prefill_solve_composite_integration.py','compose_prepared_solve_guard.py','place_prefill_solve_variants.py','run_prefill_solve_small.py','run_prefill_solve_pack.py','run_prefill_solve_integration.py','run_prefill_solve_ab.py','run_prefill_solve_candidate_profile.py','run_prefill_solve_layout.py','run_prefill_solve_variants.py','diagnose_solve_profile_cpu.py','diagnose_gdn_casts_cpu.py','rollback_prepared_solve.py','check_solve_rollback_cleanup.py','report_solve_rejection.py','nvfp4_update_runtime.py','shutdown_process_probe.py','acceptance/check_gdn_solve_pair.py','acceptance/solve_pair045_retained.py.txt','acceptance/check_gdn_rhs_pack.py','acceptance/check_gdn_prepared_integration.py','acceptance/benchmark_prefill_solve_ab.py','acceptance/profile_prefill_solve_candidate.py','acceptance/measure_rhs_layouts.py','acceptance/flat_rhs047.py.txt','acceptance/check_solve_variants.py','acceptance/qwen_ops_before_solve_processing.py.txt','acceptance/quant_profile_summary.py','acceptance/request_nvfp4_candidate.py']+[f'prompts/nvfp4-task-{n:03d}.txt' for n in range(45,54)]
for name in files:shutil.copy2(R/name,h/Path(name).name)
(h/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir() if p.name!='SHA256.json'},indent=2)+'\n');(out/'README.md').write_text('''# GDN RHS packing and paired solves — not adopted

All16accepted engine files are restored byte for byte to **ea70e47** after retaining the complete19-file experiment. No engine improvement is accepted from this package. The earlier state-tail generation failure remains quarantined separately; its implementation is excluded here.

## Real-checkpoint result

One resident native27B NVFP4 model, two ABBA blocks/four warm samples per method and length:

| Prompt tokens | Original → prepared prefill | Less time |
| --- | --- | --- |
|256|262.91 →260.71ms|0.84%|
|1,024|1,012.78 →1,010.62ms|0.21%|

The larger-prompt gain is too small to justify new kernels and solver plumbing. Generation is correctness only: both methods repeat128tokens twice at each length, all exact. Entire prefill/nextdecode logits and every cache tensor also match. Loading, tokenization and compiler startup are excluded from warm timings. These numbers must not be added to earlier speedup percentages.

## What was tested

Local045 first concatenates two already-prepared RHS tensors into one triangular solve. All86general and21captured-GDN cases are bitwise exact, but isolated two-solves83.50us versus cat/one-solve/split81.20us is only a tiny improvement. This prototype was retained outside the active engine before integration.

Local047 packs the original FP32 v*beta and ordered(k*beta)*exp products directly into a contiguous combined RHS. Its244bitwise/layout/ownership cases cover positive/zero strides, signed zeros, subnormals, finite extremes, emptybatch and an8,767-element multiplication-reassociation counterexample; metadata/inference/INT32guards pass. Shape divisors are constexpr; runtime stride tuples were accepted after review and unchanged exact tests. The isolated42.49→13.59us comparison includes the Torchcat required by the proposed paired method. The original engine's two separate solves require no such combined buffer, so this isolated result cannot establish an engine speedup.

Local048 performs one original Torch lower-unit solve on that packed RHS. Local049 body plus the exact local051 namespace-guard line supplies integration, mechanically composed by the supervisor. The corrected four-line eager RHS/two-solve fallback stays intact. All matrices, exponentials, statistics, state update and cached decode stay original. Combined298direct GDN,32dense/mixed-quantized blocks,20tiny whole models and two gradients match exactly; import/error/fallback guards pass. The real checkpoint result above also passes the stronger repeated-generation gate.

## Profile explains the result

Matched one-token profiles assert zero decode calls. At1,024tokens:

| Work | Original | Prepared |
| --- | ---: | ---: |
|Original three RHS products / new pack kernel|20.88ms|21.63ms|
|Both original solves / one prepared solve, inclusive|63.99ms|65.03ms|
|Solve kernels alone|54.81ms|49.32ms|
|All GDN copy kernels, exclusive leaf total|64.93ms|73.73ms|
|Whole GDN state scope|357.82ms|355.46ms|
|GDN kernel count|30,960|27,120|

Solve execution becomes somewhat cheaper, while copying consumes that saving. Fewer kernel launches therefore do not produce a material latency gain. Inclusive and leaf totals overlap; do not add them. Profiles include overhead and are separate from paired latency. Observed real RHS shapes/strides are retained; the value matrix has time stride1/feature stride1,024, while the normalized key has time stride6,144/feature stride1. Synthetic hot-buffer layout timings alone did not reveal a useful improvement.

Local053 then compares column pairing, column-major pairing, leading-batch pairing and column-major leading-batch pairing. All408comparisons are bitwise exact, including16chunks captured from a synthetic full dense GDN block. End-to-end isolated pack/solve/split timings, each against its alternating two-solve baseline:

| Variant | Original → variant |
| --- | --- |
|columns|82.65 →86.75us|
|columns_fortran|82.76 →84.65us|
|batch|83.23 →100.76us|
|batch_fortran|83.25 →102.85us|

None beats the original baseline. These are synthetic component diagnostics, not downloaded-checkpoint validation. Their outputs/layouts, flags and all failures are recorded.

## Provenance, rollback and next step

All implementation came from local Qwen. Raw046 missed metadata guards;049 had an overly broad import prefix;050 proposed the wrong guard patch;052 had a column-major shape error. Each was retained/rejected before placement or composition. Corrected047/048/049+051/053 source and before/after files are retained. The supervisor authored only tests, diagnostics, orchestration, mechanical source placement and exact rollback. No numerical tests were weakened.

Candidate quality/protected acceptance suites were not run because the experiment was not adopted for performance. The restored accepted source retains its previous40quality/446NLL and56Qwen/15Llama validation; this is not a new claim about the rejected candidate. All19experimental files were retained before restoring all16baseline hashes. Containers stop0/OOMfalse, GPUidle, serviceMainPID0, sameboot, no newSwap, STOP/holds retained; see cleanup-check.json.

Model: unsloth/Qwen3.8-27B-NVFP4, revision f0b7c9e722f5565102fff8481c99e4d86ae099c7, pinned Torch2.13+cu130 runtime. Main run '''+str(run)+'''. Weights, compiler caches and large traces remain outside Git.

The next smaller measured target is the eager full-value BF16→FP32 copy,7.39ms at1,024tokens. A separate local-model task will test deferring it until the existing FP32 multiplication; output and carried state must remainFP32. This package does not implement that change.
''');print(out/'README.md')
