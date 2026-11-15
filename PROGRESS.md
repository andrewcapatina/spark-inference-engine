# Overnight prototype progress

Experiment prepared on November 13, 2026. Stopped at the user's request for
Astra review before proceeding; both experiment containers are stopped.

## Setup verified

- An isolated OpenClaw 2026.7.2 container has the project mounted at `/workspace`.
- Pinned Python, PyTorch CPU, Transformers, Safetensors, NumPy, and pytest are
  available; importing the Llama reference implementation succeeded.
- The offline SmolLM2-135M-Instruct checkpoint loaded successfully, with
  134,515,008 parameters. Its snapshot revision is recorded in the supplied
  `reference-manifest.json`.
- A separate Ollama 0.30.0 container serves the downloaded DavidAU 27B Q8_0
  model with the Qwen3.5 chat/tool renderer.
- The model returned a structured tool call. An OpenClaw smoke turn then ran
  a Python command, wrote a project marker, and read it back successfully.

## Implementation

The first coding turn was interrupted before completion. No prototype changes
were present in the worktree when the run stopped. The setup and logs are
preserved. Do not resume until the user requests it after review.

Review `OVERNIGHT_TASK.md` and the controller/configuration under
`/home/cap/spark-inference-overnight-20261113/` before restarting. The existing
8:00 AM stop target must be checked against the intended resumed run time.

Controller state and logs: `/home/cap/spark-inference-overnight-20261113/`.
The controller saves per-turn logs and checkpoints and records test exit codes.
Replace this setup note with evidence of actual work as the experiment proceeds.

## Authorized minimum run
User requested resumption. Fixture paths and independent acceptance contract are corrected. Start stage 1 and follow controller feedback.

## Stage 1: RMSNorm (completed)
- Implemented `spark_engine/ops.rms_norm` with manual math (no nn.RMSNorm).
- Added 6 unit tests covering shape, dtype, unit/weighted scaling, noncontiguous input, zeros, and input preservation.
- Command: `python3 -m pytest -q tests/` → 6 passed in 0.45s.
- Next: Stage 2 — Llama forward pass and local weight loading.

## GPU runtime requested
User requested CUDA execution for the prototype. Runtime now uses the existing
source-built PyTorch CUDA 13.0 image. Acceptance checks require CUDA tensors.
The partial stage-2 model.py is preserved; it still needs independent validation.

## Short-turn loop
User requested one fix, one targeted check, then checkpoint. Coding turns now
have a 300-second limit. Current target: RoPE broadcasting in tiny forward.
Preserved loader/RoPE draft; tiny-forward baseline still has 2 failures.

## Verified forward baseline (direct repair)
- Repaired RoPE rank/frequency ordering, duplicate GQA expansion, and untied output-head mapping.
- GPU acceptance: 9 passed in 2.59s, including tiny tied/untied parity, real SmolLM2 parity, nonzero-position RoPE, and batched causal masking.
- Existing unit tests: 6 passed in 0.50s.
- Reference loading now explicitly starts on CPU before moving to CUDA; this avoids requiring accelerate for global-device dispatch. All forward comparisons execute on CUDA.
- Next local-model task: Attention.forward KV caching only. Follow LOCAL_TASK_GUIDE.md.
- Controller now stops for review after two identical failures, three turns without newly passed checks, unrelated function edits, or any forward-baseline regression.

## 2026-11-13 10:00 UTC — Stage 3: Attention KV cache
- Extended Attention.forward with past_key_value/use_cache; RoPE on new q/k only, concat old/new k/v, store unrepeated KV, expand for attention math.
- pytest -k test_attention_kv_cache PASSED.
- Next: test_incremental_cache_and_request_isolation (DecoderLayer + CausalLM prefill/decode).

## 2026-11-13 10:05 UTC — Stage 3: Incremental cache and request isolation
- Extended DecoderLayer.forward with cache args; added CausalLM.prefill and CausalLM.decode with per-request cache list and position offset from past keys.
- pytest -k test_incremental_cache_and_request_isolation PASSED.
- Next: test_greedy_generation_and_stop (CausalLM.generate loop).

## 2026-11-13 10:14 UTC — Stage 3: Decoder return-contract repair
- Fixed DecoderLayer.forward to conditionally unpack Attention result only when use_cache=True, restoring tensor-only return for ordinary forward.
- pytest -k test_decoder_return_contract PASSED (batch sizes 1 and 2).
- Next: test_greedy_generation_and_stop.
Stage 3: added spark_engine/__main__.py CLI with argparse, AutoTokenizer(local_files_only=True), CausalLM.generate, --device cuda; test_cli_runs_offline now passes.

## 2026-11-14T08:16:32-08:00 — Qwen CLI (directly authorized by user)
- Added spark_engine/cli.py and routed python -m spark_engine through it. Qwen Safetensors/GGUF auto-detection, interactive history, system prompt, /reset, /exit, EOF and Ctrl-C cancellation; original Llama completion behavior preserved. Native model/math/loader modules unchanged.
- Validation: python3 -m unittest discover -s tests -p test_cli.py -v: 16 passed. Existing CUDA acceptance/test_contract.py: all 15 passed. Actual 9B and 27B CLI subprocesses each passed two conversational turns plus reset/repeat with one model load per session.
- Evidence: /home/cap/spark-inference-overnight-20261113/feedback/cli-validation.json; logs: /home/cap/spark-inference-overnight-20261113/runs/cli-20261114T081009. Container restored to stopped; overnight controller not restarted.
- Usage: CLI_USAGE.txt. Greedy generation; replies appear after completion. Token streaming and sampling remain future work.

## 2026-11-14T10:52:59-08:00 — CLI checkpoint and architecture walkthrough
- Checkpointed the directly authorized CLI in a9e75e6 and configured origin as https://github.com/andrewcapatina/spark-inference-engine.git. Remote inspection found no existing refs.
- Added README.md and ARCHITECTURE.md with the implemented request flow, source map, model loaders, hybrid layers, cache ownership, memory behavior, validation scope and unmeasured profiling candidates. Preserved INFERENCE_ENGINE_DIAGRAM.txt byte-for-byte as the original roadmap.
- Checks: 16 CLI unit tests pass; documentation links/anchors and checkpoint dimensions verified; native inference source hashes still match the validated engine; git diff --check passes. Existing CUDA/real-model validation is cited, not rerun for documentation. Git history contains no model-weight files.
- No inference changes or benchmark runs in this step. Next phase can measure the documented profiling candidates.

## 2026-11-14T11:09:53-08:00 — Performance harness, before full-model runs
- Added tools/benchmark_qwen.py with synchronized uninstrumented repetitions, separate CUDA-event diagnostics, memory accounting, fixed token workloads and separate bounded profiler capture. BENCHMARKS.md defines metric boundaries and cache-state limits. Raw runs/traces ignored by Git.
- Seven accounting/isolation tests pass. Existing tiny native CUDA Qwen probe preserves outputs under event observation and profiling; profiler GPU timing is available on installed PyTorch2.11.0a0/CUDA13.0/GB10. No inference source changes.
- User closed the active27B chat; process inspection confirmed no active model session and inactive controller. Full9B/27B benchmark matrix is next, sequentially; no baseline performance result claimed yet.

## 2026-11-14T19:44:58.881033+00:00 — Benchmark interrupted by host reset
- Full9B32/256/1024-token matrix and profile completed. 27B32/256 cases completed; host reboot interrupted the next case before the27B profile. Completed data and an explicitly partial report are preserved under benchmarks/results/2026-11-14-interrupted/.
- Six saved27B repetitions show stable CUDA allocated/reserved memory and zero process swap; no latest-crash OOM journal entry found. Root cause remains unproven; driver messages and missing final-phase telemetry recorded. Container stopped after host reset, OOMKilled=false, exit255.
- No heavy retry or inference changes. Next: confirm reboot circumstances, add durable telemetry, and isolate a bounded non-profiler diagnostic before resuming the full benchmark.

2026-11-14T19:50:40.047536+00:00 | Shutdown recovery: user confirms machine required powering on again, resembling prior OOM events. Memory exhaustion is a working hypothesis, not proven leak. Harness-only patch adds optional CUDA allocator cap, between-stage host headroom check, and fsync phase/sample/results logging. `python3 -m unittest discover -s tests -p test_benchmark.py`: 7 pass; validated native source hashes unchanged. No GPU workload restarted; cap/headroom behavior on GPU remains unverified. Next: validate safeguards with a tiny allocation probe, then a separately labeled bounded diagnostic without profiling.

2026-11-14T20:17:29.704218+00:00 | Bounded recovery diagnostics PASSED. Tiny real-CUDA cap probe allows1MiB/rejects32MiB under16MiB limit. Actual27B no-profiler tests:512/1024 prompt with8 new tokens, then1024 prompt with128 new tokens,1 warmup+3 timed repeats+separate events. 56GiB allocator cap; external0.5s host watchdog and durable logs. Full repetitions match outputs, allocated50.106GiB/reserved52.477GiB/peak51.732GiB identical; minimum host available49.30GiB, zero swap. Median31.619s/4.048tokens/s. Native hashes unchanged, same boot, no matching kernel warnings. Worker exit0; deliberate container stop leaves init exit137/OOMKilled=false; both containers stopped. Seven benchmark unit checks pass. Original shutdown NOT reproduced or explained; profiler and long-session behavior remain unvalidated. Evidence benchmarks/results/2026-11-14-memory-diagnostics/README.md. Next: keep profiling isolated and use caps for further controlled runs.

2026-11-14T20:23:32.906427+00:00 | Consolidated benchmark findings in PERFORMANCE.md, with source-checked tables for original9B/partial27B and separately capped27B recovery, fresh-process load/cache caveats, GPU-event latency definitions and profiling limits. Selected first local-Qwen task: remove final unused decode in QwenCausalLM.generate, directly measured at~93ms9B/~233–239ms27B; expected~0.7–0.8% of128-token interval, not claimed achieved. OPTIMIZATION_TASK.md specifies function-only authorship, observer-accounting repair, unchanged56Qwen/15Llama gates and capped matched before/after tests. Task prepared, not queued/executed; no engine or GPU work this turn. Validation: report values derived from saved JSON, native hashes/output consistency checked, documentation links/diff checked. Next: supervisor repairs measurement accounting, then localQwen implements the one scoped optimization.

2026-11-14T20:29:56.993084+00:00 | Started authorized final-decode optimization. Supervisor repaired event accounting only: actual returned token count, final return marker when decode is absent, accurate unused-decode absence;10 portable harness tests pass. Deterministic native generation budget fixture and tiny native CUDA observer checks pass on baseline, including one-token and early EOS, without profiler. Experiment locks,56/24GiB model allocator caps and external host watchdog prepared. Native engine source remains unchanged. Run optimization-final-decode-20261114T202920Z; matched before measurements next, then localQwen-only candidate.

2026-11-14T20:56:42.218632+00:00 | Candidate authored entirely by localQwen; retained637-byte method/response and verbatim scope-guard placement. Focused CUDA budget/observer tests pass. Aggregate regression was deliberately stopped by the host watchdog after32 observed passes when cumulative host swap exceeded1GiB; at abort45.08GiB remained available, no assertion failure or host reboot. Original partial logs preserved. Recovery isolates each actual checkpoint in a fresh process and reruns all56 unique Qwen gates, plus head-layout/import interaction checks and15Llama cases; GPU cap unchanged56GiB, external host floor tightened32GiB, swap budget measures new growth per command rather than stale global total. No numerical tolerances or engine implementation changed for recovery. Verification remains pending.

2026-11-14T21:03:28.769472+00:00 | Correctness recovery complete:52 small Qwen gates + all4 actual-checkpoint gates pass in isolated processes; head-layout fixture is also paired with actual9B conversation and actual27B greedy checks to retain import-isolation coverage. Audited union56 uniqueQwen gates and15Llama cases; repository budget test passes6 cases, focused CUDA observer/budget checks pass. No numerical expectation/tolerance changed. Candidate637-byte response/model/source provenance retained in benchmarks/results/2026-11-14-final-decode. Native method authored entirely by localQwen and committed18894a8 with Local Qwen author; supervisor only placed its exact retained text. Matched post-change measurements pending.

2026-11-14T21:25:41.278849+00:00 | SECOND HOST REBOOT CONFIRMED during after27B1024/128 timed repetition0, after completed warmup. Last durable host snapshot59.26GiB available/no new swap growth; phase14.25GiB CUDAfree,50.106allocated/52.504reserved, warmuppeak51.732under56cap. No tied OOM/GPUfault persisted; cause unproven. Guards did not prevent shutdown. Both containers/controller stopped; GPU_WORKLOAD_HOLD and STOP recorded, experiment runners fail fast. All56Qwen+15Llama correctness gates previously passed at localQwen18894a8, but performance comparison is7/8 cases only: short cases~5–8% lower latency, saved128-token cases slower. No blanket speedup claim; paired diagnostic NOT executed. Original raw data preserved, partial report/checkpoint updated. Next: review reliability evidence and user-selectable options; no automatic heavy restart or next optimization.

2026-11-14T21:32:26.178654+00:00 | User requires one large model at a time. Audited scheduling: second crash had only native27B worker scheduled,9B worker exited and coder server stopped; Qwen references are saved tensors/tokenizers, not a second full reference model. Earlier aggregate regression shared one process across model tests, allowing retention risk. Harness now refuses a coding server during native commands, refuses existing Python workers before each command, rejects aggregate multi-checkpoint regression, and restarts the agent container at each large-checkpoint boundary. Seven CPU-only guard tests pass, probe source compiles; no GPU runtime validation or restart, crash hold retained. Native implementation unchanged. Evidence model-isolation-audit.json; next remains reliability/options review.

2026-11-14T23:01:51.791048+00:00 | User authorized cleaner-shutdown trial. Preserved original stopped agent as backup, created identical-image/settings container with init and signal-aware keepalive; no engine changes. Empty CPU-only lifecycle passes: stop0.112s,exit0,pid0,OOMfalse. Global GPU_WORKLOAD_HOLD remains; dedicated one-shot trial prepared for9B only,256prompt/8outputs,1warmup+3samples+events,24GiB allocator cap,64GiB host floor/256MiB new-swap guard. Added optional benchmark clean release after timing,10 benchmark unit tests pass. Next: execute one bounded trial and verify process/container/allocator/host-memory recovery; no27B run or new optimization selected.

2026-11-14T23:12:27.299813+00:00 | Clean-shutdown9B bounded trial PASSED. First attempt conservatively stopped due to counting timeout wrapper as second Python worker; saved inventory showedone, parser fixed and4 checks pass, separate retry preserved. Actual9B256prompt/8outputs,1warmup+3samples+events: median808.82ms, all output hashes match earlier accepted-source case,7decodecalls for8tokens. Explicit cleanup8.125MiB allocated/40MiB reserved; worker exit0, freshCUDAprobe0/0 and noNVML compute processes; container stop0.165s/exit0/pid0. Minimum host available77.54GiB, zero swap growth, same boot. Host available recovered115.35->114.59GiB, butOS cache3.33->20.11GiB andCUDAfree109.96->92.33GiB; not proof of exact cache attribution or a shutdown fix. No27B trial; native source unchanged; globalhold retained. Evidence benchmarks/results/2026-11-14-clean-shutdown. Next: review memory/cache headroom before choosing further GPU work.

2026-11-14T23:21:18.415302+00:00 | Explicitly requested27B-only clean-shutdown trial PASSED. Native unchanged;256prompt/8output,1warmup+3samples+events,56GiB cap/32GiB host floor/256MiB new-swap guard. Load141.408s, median2.098s(3.812outputtokens/s inclprefill); firsttokenCUDA508.37ms; usefuldecode4.392tokens/s; peakallocation52.47GiB includingload. Outputhash matches prioroptimizedcase,7decodecalls/8tokens. Minimumhostavailable49.55GiB, zero newswap. Explicitrelease8.125MiB/20MiB; worker0, NVMLempty/freshCUDA0/0, containerstop0.164s/exit0/pid0,sameboot. Hostavailable recovered within0.78GiB; OSfilecache remains. Bothsmallmodelcleanuptrials nowpass, but prior27B1024/128 crash unresolved/fullbaselineincomplete. Evidence benchmarks/results/2026-11-14-clean-shutdown-27b; no further GPU job or optimization selected.

2026-11-14T23:45:00.336758+00:00 | THIRD HOST POWEROFF: user confirms spontaneous power-off with original adapter/cable. Larger27B trial first256/128 passed3samples median29.799s plus verified clean teardown. Unused9B file-cache advice restoredCUDAfree65.41->82.11GiB without changingfiles; prompt lengths isolated in fresh processes, one model, no profiler.1024/128 warmup and timedrep0(31.800s, exactpriorhash) completed; host poweredoff just afterrep1started. Last hostavailable59.07GiB/no swap; beforefailedrepCUDAfree26.29GiB/allocated50.106/reserved52.477; GPUlast79°C/84.45W, not proof of thermaltrip. Kerneljournal has no tied OOM/Xid/orderedshutdown. Newboot761dc008; agent255/OOMfalse, bothcontainers/controllerstopped. Originaldataretained; performanceincomplete; allGPUandlargerretryholds active. Native source unchanged. InstalledNVIDIA safe-mode collector requiresroot; fielddiag absent/notinstalled. Next: root diagnostic collection and reliability review, notanother heavy retry or optimization.

2026-11-14T23:46:03.603983+00:00 | Physical confirmation and diagnostic handoff: user reports spontaneous power-off and original NVIDIA adapter/cable. NVIDIA collector exists at/usr/bin/nvidia-bug-report.sh; inspected installed flags and root guard. Prepared local safe-mode collector plus kernel journal/pstore-copy script, shell syntax checked, NOT executed because root is required. No fielddiag package installed or hardware stress launched; no packages/settings changed. All authorized unprivileged evidence packaged; GPU holds retained. Next requires user-run sudo collection, then review.

2026-11-14T23:52:44.942298+00:00 | User completed root diagnostic collector: exit0, rawreport private, pstore copy clean/empty, no recorded latest-crash Xid/OOM/panic. Read-only fwupd inventory confirms EC0x03000508(3.5.8); DMI UEFI R:1.110.13 vs official currentEC3.13.7/UEFI1.111.17. OSOTA7.6.0,driver580.178.04,kernel7.0.0-1019 match currenttable. Related NVIDIA-staff thread notes firmware sensor improvements, not a diagnosis for this unit. User reports Dashboard firmware update available. Interrupted cached get-updates query produced no file/process; no firmware/system update command issued. Recommend official Dashboard update and prompted reboot, then verifyversions before anyload. GPUholds persist/native unchanged. Sanitized review in third-crash DIAGNOSTIC_REVIEW.md; fullbundle remains outsideGit.

2026-11-15T00:19:57.439277+00:00 | POST-UPDATE REAL27B HEAVY TRIAL PASSED |1024prompt/128output,1warmup+3timed+events,one model,no profiler. Median31.417s/4.074tokens/s; all hashes match previous native,127decodes. PeakCUDA52.47GiB,minhost46.56GiB,no swap; GPUmax81C/ACPImax88.4C. Worker0,freshCUDA0/0,NVMLempty,container0/stop0.114s,sameboot,source unchanged. Verified updated EC3.13.7/SoC2.158.15/PD0.5.25 and kernel1025; driver unchanged. No tied kernel fault. Prior cause unproven; onepass not long-session validation. Bothcontainers/controllerstopped; holds retained. Evidence benchmarks/results/2026-11-14-post-update-27b-heavy. Next: review result/options before further work.

2026-11-15T00:21:27.179531+00:00 | Cross-run performance review: seven matched cases show eight-output latency 4.95–7.89% lower, but saved128-output cases1.34–2.40% slower. Latest27B1024/12831.417s versus capped pre-change31.619s (~0.64% lower) is confounded by stack/runtime/cache changes; no blanket speedup or firmware-only attribution. Added benchmarks/results/SUMMARY.md covering all trial families, excluded load-time comparisons and incomplete cases. Source JSON reviewed; no engine changes or GPU runs. Next proposed only: bounded same-loaded-model alternating comparison, not launched.

2026-11-15T00:34:32.710633+00:00 | SAME-LOADED27B PAIRED COMPARISON PASSED |256prompt,8/128output,4AB/BApairs each+warmups.8outputs2.338->2.112s(9.66%lower);128outputs29.808->29.573s(0.79%lower,4.328outputtok/s inclprefill). All8pairs favorcurrent; exacthashesmatch; no enginefileedits. Restoredoldmethod runtimebinding; unchangedsourcehashes. No swap/reboot,worker0/freshCUDA0/0/NVMLempty/container0/hostrecovered. Initialharnesspathfailurebeforeload retained/corrected. Existingpatch retained; no nextoptimizationselected. Evidence benchmarks/results/2026-11-14-paired-27b. Usernowasks externalPyTorch/vLLMcomparison; externalmetrics mustbe labeledquantized/differentworkloads.

2026-11-15T00:36:16.175639+00:00 | NVFP4 planning review: inspected Unsloth Qwen3.8-27B-NVFP4 model card/config and official vLLM Spark guidance. Candidate is same27B dense hybrid family, mixed FP8/NVFP4 compressed-tensors, not a drop-in native format and not identical weights to existing DavidAU fine-tune. Recommend pinned checkpoint, one-model external baseline on validated sm121 runtime, then local-Qwen-only native packed-weight/kernels work with small operator references and unchanged BF16 gates plus separate quantization-quality tests. No weights downloaded, packages/images installed, queue resumed or model loaded. Sources https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4 and https://vllm.ai/blog/2026-06-01-vllm-dgx-spark .

2026-11-15T00:43:05.460549+00:00 | NVFP4 DOWNLOAD COMPLETE |unsloth/Qwen3.8-27B-NVFP4 revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 at /mnt/nas-aimodels/huggingface/unsloth/Qwen3.8-27B-NVFP4.13files/23444511852bytes; all3LFS SHA256 and10GitblobSHA1 match published pinned manifest, diskreadback verified before atomic finalization. Includes model,optionalMTP,tokenizer/config/templates. No package/driver/container changes or inference/model load. Currentengine cannot execute this mixedNVFP4/FP8 checkpoint: localQwen must implement packed-weight/scaling loader and compatible quantizedGPU operations; downloading alone is not runtime support. Existing BF16 numerical gates remain unchanged. Metadata benchmarks/model-manifests/qwen38-27b-nvfp4.json. Next: reference baseline/implementation plan, not automatically launched.

2026-11-15T00:53:06.369984+00:00 | ACTUAL NVFP4 VLLM REFERENCE PASSED |256/1024prompts,128outputs,1warmup+3samples each,11.453/11.093tok/s inclprefill,allrepeat IDsmatch. FlashInferCUTLASSNVFP4+FP8, eager/no graphs/no MTP, single model; teardown0/probe0/0/hostrecovery/sameboot. Nativeengine unchanged; newcheckpoint+runtime confound oldBF16 comparison. Evidence benchmarks/results/2026-11-14-nvfp4-vllm. User authorized nativequant update next, localQwen authors all implementation.

2026-11-15T01:07:03.136887+00:00 | NATIVE NVFP4 OPERATOR TINY-CUDA MILESTONE |LocalQwen response003 module47lines passed independent unpacked matmul/global-scale/layout/storage tests; priorresponses001/002 retained rejected. Not fullcheckpoint/loader/generation support. FP8response004 signed-max rejected,005 retained after actual scaled_mm stride failure;006 localQwen repair inprogress. No supervisor enginecode written. Existing BF16 source unchanged; full acceptance rerun pending integration. vLLM actualreference complete11.453/11.093tok/s. Evidence benchmarks/results/2026-11-14-native-quant-components.

2026-11-15T02:00:24.295779+00:00 | REAL NATIVE NVFP4 FUNCTIONAL CHECKS PASSED |LocalQwen modules c491ef7/c2d74ce/c207950, dispatch56c6f08, chatcompatac4cbc0. Actual27B168NVFP4+233FP8linears,22.586GBresidenttensors(~21.04GiB), exactfirst8tokensmatchvLLM256/1024prompts,2turnmath4then7. Firstchat failure was newHF BatchEncoding default, localQwen return_dictFalse fix passedCPUold/new andactualchat. Parentcleanupguard rejected72MiBlibraryworkspace, not functionaltestfailure; tiny no-model control reproduces72MiB and explicitcuBLAS+FlashInfercacheclear restores0/0. Harness usesweakmodeltensorrefs+same16/64MiBcleanupbounds;11portablebenchmarktests pass. Nativeperformance/protectedBF16regression pending, not generalquality/longcontext validation. NativeC/modelserverstopped before protectedregression starts; allnumericalgates unchanged.

2026-11-15T02:08:42.353742+00:00 | NATIVE QUANTIZATION CHECKPOINT |Real generation+chat functionalpassed, cleanup72MiB attributedby no-modelcontrol tocuBLAS/FlashInferworkspace and explicitcleanup0/0; unchanged16/64MiBbounds plusweakmodelrefs in measurementharness.11benchmarktests pass. Budget+52small+real9Bprotectedpassed; remainingprotectedreal/Llama running. Model-specific preflight56GiB9B/80GiB27B,32GiBhostfloor/256MiBnewSwap, one checkpoint perprocess; numerical gates untouched. Detailedprovenance/rejectedpatches/fixtures in benchmarks/results/2026-11-14-native-nvfp4. Nativebenchmarkpending.

2026-11-15T02:48:45.501195+00:00 | NATIVE NVFP4 PACKAGE COMPLETE, ACCURACY PARTIAL | LocalQwen engine only; native256/1024prompt128output9.171958/8.346411tok/s,21.04GiB resident,56Qwen+15Llama protectedpassed unchanged. Actual benchmark cleanup0/0 and no surviving model tensors; all4experimentcontainers stopped/serviceinactive. Full128IDmatch vLLM1024;256 later-token divergence persists with BF16KVreference (attentionbackend also changes), cause unresolved. Next identical-input logit/teacher-forced reference validation before furtheroptimization. Report benchmarks/results/2026-11-14-native-nvfp4-benchmark/README.md.
Commands: guarded native tools/benchmark_qwen.py --model /model --prompt-lengths2561024 --new-tokens128 --warmups1 --repetitions3 --cuda-limit-gib32 --min-available-gib32 --clean-release (actual argv in trial-status.json); diagnostic /harness/diagnose_vllm_bf16_kv.py matched-kv-reference-20261115T024116Z; both exit0. docker inspect confirms stopped PID0/OOMfalse for all4experimentcontainers; systemctl user service inactive MainPID0. No new engine changes after localQwen ac4cbc0.
Final portable verification: python3 -m unittest discover -s tests -p test_benchmark.py -v:11 passed; -p test_cli.py -v:16 passed. git diff --check clean. Every engine SHA256 still equals benchmark source. Archived fixture renames verified byte-identical to committed originals; acceptance fixtures unchanged.
