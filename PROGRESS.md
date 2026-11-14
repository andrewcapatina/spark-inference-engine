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
