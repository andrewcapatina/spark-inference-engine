# Inference-engine prototype workspace

Work only on this project's inference-engine prototype and its documentation.
The surrounding infrastructure ideas are future work. Read QWEN_TASK.md and
run_feedback/latest.txt for the current phase. ENGINE_CONTRACT.md describes the
completed Llama baseline; preserve it. Reference files are mounted
read-only at /workspace/reference/smollm2; the design document is
/workspace/INFERENCE_ENGINE_DIAGRAM.txt. These paths work with the file tools.

Implement, run checks, and record evidence in PROGRESS.md. Prefer small modules
with explicit tensor shapes and clear state ownership. The small Llama baseline is complete. The current task is Qwen hybrid support
under QWEN_TASK.md; its queue and file scopes govern this phase.

Do not change the existing component diagram's meaning or remove source notes.
Do not replace the engine with a call to Transformers.generate, Ollama, vLLM,
or another serving engine. Those are references/backends for the coding agent.
Use PyTorch CUDA tensor operations on the GB10 for the first implementation; custom kernels and
multi-Spark execution are later milestones.

Keep model weights, caches, and generated bulk output out of Git. Do not access
unrelated projects, accounts, messaging services, or host administration. Do not
install packages or fetch arbitrary code during this run; needed dependencies
and a small reference checkpoint are supplied. Do not spawn other agents.

At the end of each turn, update PROGRESS.md with completed work, actual commands
and results, remaining failures, and the next step. Keep progress reproducible.

Short-turn protocol: one focused fix (at most 50 changed lines), one targeted
check, a brief progress note, then finish. The controller checkpoints each turn
and chooses the next failure. Avoid investigating multiple issues in one turn.

The forward baseline is verified. For Qwen follow QWEN_TASK.md and the named
queue task. LOCAL_TASK_GUIDE.md applies only to the completed Llama phase.
Only edit the functions that task names; the controller checks function changes
and runs forward regression checks every turn. A repeated failure stops for review.
