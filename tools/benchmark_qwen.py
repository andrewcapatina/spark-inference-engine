"""Observe native Qwen inference without changing its implementation.

Baseline repetitions call the unmodified generate method. CUDA-event method
observation and torch.profiler run separately and are explicitly diagnostic.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain how a computer turns a question into an answer. Discuss memory, arithmetic, and clear examples.\n"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def proc_memory():
    result = {}
    for filename, keys in (("/proc/self/status", ("VmRSS", "VmHWM", "VmSwap")),
                           ("/proc/meminfo", ("MemAvailable", "SwapFree"))):
        for line in Path(filename).read_text().splitlines():
            key, _, value = line.partition(":")
            if key in keys:
                result[key + "_bytes"] = int(value.split()[0]) * 1024
    return result


def memory(torch):
    free, total = torch.cuda.mem_get_info()
    return {"cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_global_free_bytes": free, "cuda_total_bytes": total,
            **proc_memory()}


def source_info():
    paths = list((ROOT / "spark_engine").glob("*.py")) + [Path(__file__)]
    result = {"sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


@contextmanager
def observe(engine, factory):
    """Restore instance attributes exactly, including absence of an override."""
    previous = {name: (name in vars(engine), vars(engine).get(name))
                for name in ("prefill", "decode")}
    try:
        for name in previous:
            setattr(engine, name, factory(name, getattr(engine, name)))
        yield
    finally:
        for name, (existed, value) in previous.items():
            if existed:
                setattr(engine, name, value)
            else:
                delattr(engine, name)


def event_metrics(start, finish, prefills, decodes, selections):
    count = len(selections)
    window = selections[0].elapsed_time(selections[-1]) if count > 1 else None
    return {"total_cuda_ms": start.elapsed_time(finish),
            "prefill_cuda_ms": sum(a.elapsed_time(b) for a, b in prefills),
            "first_token_cuda_ms": start.elapsed_time(selections[0]) if count else None,
            "decode_call_cuda_ms": [a.elapsed_time(b) for a, b in decodes],
            "observed_decode_calls": len(decodes),
            "selected_tokens": count,
            "decode_token_window_cuda_ms": window,
            "decode_tokens_per_second_cuda": (count - 1) * 1000 / window if window else None,
            "final_unused_decode_cuda_ms": decodes[-1][0].elapsed_time(decodes[-1][1]) if decodes else None}


def timed_diagnostic(torch, engine, ids, count):
    event = lambda: torch.cuda.Event(enable_timing=True)
    prefills, decodes, selections = [], [], []

    def factory(name, original):
        def call(*args, **kwargs):
            if name == "decode":
                selected = event()
                selected.record()
                selections.append(selected)
            start, finish = event(), event()
            start.record()
            output = original(*args, **kwargs)
            finish.record()
            (prefills if name == "prefill" else decodes).append((start, finish))
            return output
        return call

    torch.cuda.synchronize()
    start, finish = event(), event()
    with observe(engine, factory):
        start.record()
        generated = engine.generate(ids, max_new_tokens=count, eos_token_id=None)
        finish.record()
        torch.cuda.synchronize()
    result = event_metrics(start, finish, prefills, decodes, selections)
    result["prefill_tokens_per_second_cuda"] = ids.shape[1] * 1000 / result["prefill_cuda_ms"]
    result["output_sha256"] = output_digest(generated, ids.shape[1])
    del generated
    return result


def output_digest(generated, prompt_length):
    return hashlib.sha256(generated[:, prompt_length:].detach().cpu().numpy().tobytes()).hexdigest()


def summarize(samples):
    wall = [row["wall_ms"] for row in samples]
    return {"repetitions": len(samples), "median_wall_ms": statistics.median(wall),
            "min_wall_ms": min(wall), "max_wall_ms": max(wall),
            "median_generated_tokens_per_second": statistics.median(
                row["generated_tokens_per_second"] for row in samples),
            "peak_cuda_allocated_bytes": max(row["memory"]["cuda_peak_allocated_bytes"] for row in samples),
            "peak_cuda_reserved_bytes": max(row["memory"]["cuda_peak_reserved_bytes"] for row in samples),
            "repeated_outputs_match": len({row["output_sha256"] for row in samples}) == 1}


def profile_request(torch, engine, ids, count, output):
    from torch.profiler import ProfilerActivity, profile, record_function

    def factory(name, original):
        def call(*args, **kwargs):
            with record_function("native." + name):
                return original(*args, **kwargs)
        return call

    # Warm this particular shape outside the profiler.
    warm = engine.generate(ids, max_new_tokens=count, eos_token_id=None)
    del warm
    torch.cuda.synchronize()
    with observe(engine, factory):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True, with_stack=False) as prof:
            generated = engine.generate(ids, max_new_tokens=count, eos_token_id=None)
            torch.cuda.synchronize()
    digest = output_digest(generated, ids.shape[1])
    del generated
    prof.export_chrome_trace(str(output / "profile-trace.json"))
    operators, stages = [], []
    for event in prof.key_averages():
        row = {"name": event.key, "calls": event.count,
               "self_cpu_ms": event.self_cpu_time_total / 1000,
               "self_device_ms": event.self_device_time_total / 1000,
               "total_cpu_ms": event.cpu_time_total / 1000,
               "total_device_ms": event.device_time_total / 1000,
               "self_cpu_memory_bytes": event.self_cpu_memory_usage,
               "self_device_memory_bytes": event.self_device_memory_usage}
        if event.key.startswith("aten::"):
            operators.append(row)
        elif event.key.startswith("native."):
            stages.append(row)
    operators.sort(key=lambda row: row["self_device_ms"], reverse=True)
    return {"prompt_tokens": ids.shape[1], "new_tokens": count,
            "output_sha256": digest, "stages": stages,
            "operator_self_device_ms_sum": sum(row["self_device_ms"] for row in operators),
            "operators": operators[:30], "trace": "profile-trace.json",
            "timing_scope": "Profiled diagnostic only; not baseline latency"}


def run(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from spark_engine.cli import _load_qwen, _model_kind

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    if args.cuda_limit_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.cuda_limit_gib * 1024 ** 3 / total
        if not 0 < fraction <= 1:
            raise ValueError("CUDA limit must be positive and no larger than device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction)
    def checkpoint(phase, **details):
        row = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase,
               "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               "memory": memory(torch), **details}
        with (args.output_dir / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if row["memory"]["MemAvailable_bytes"] < args.min_available_gib * 1024 ** 3:
            raise MemoryError("System available memory below diagnostic headroom threshold")
    context_init_ms = (time.perf_counter() - started) * 1000
    result = {"schema_version": 1, "started": datetime.now(timezone.utc).isoformat(),
              "model": str(args.model.resolve()), "source": source_info(),
              "settings": {"cuda_limit_gib": args.cuda_limit_gib,
                           "min_available_gib": args.min_available_gib,
                           "prompt_lengths": args.prompt_lengths, "new_tokens": args.new_tokens,
                           "repetitions": args.repetitions, "warmups": args.warmups,
                           "batch_size": 1, "dtype": "bfloat16", "eos_enabled": False,
                           "prompt_recipe": PROMPT, "prompt_kind": "repeated raw-text token IDs, truncated to exact length"},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                              "interop_threads": torch.get_num_interop_threads(),
                              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_benchmark": torch.backends.cudnn.benchmark},
              "methodology": {"load": "Fresh process/model; OS/NAS cache state uncontrolled, not guaranteed disk-cold",
                              "baseline": "Unmodified native generate; wall timing synchronized before/after, input IDs already on CUDA",
                              "events": "Separate observed run; first-token and decode throughput use CUDA stream event markers, exclude text I/O",
                              "memory": "CUDA peaks per phase include weights; VmHWM is process-lifetime peak RSS; global free memory is shared, do not sum capacities",
                              "work": "Fixed output length with EOS disabled; performance workload, not response-quality validation"},
              "cases": [], "status": "loading"}
    result_path = args.output_dir / "results.json"
    write_json(result_path, result)
    result["cuda_context_init_ms"] = context_init_ms
    result["before_load_memory"] = memory(torch)
    torch.cuda.reset_peak_memory_stats()
    checkpoint("before_load")
    started = time.perf_counter()
    text = _load_qwen(args.model, _model_kind(args.model, "auto"), "cuda")
    torch.cuda.synchronize()
    result["model_tokenizer_load_ms"] = (time.perf_counter() - started) * 1000
    result["loaded_memory"] = memory(torch)
    result["parameter_bytes"] = sum(p.numel() * p.element_size() for p in text.engine.parameters())
    result["status"] = "benchmarking"
    write_json(result_path, result)
    print("MODEL_LOADED " + json.dumps({"load_ms": result["model_tokenizer_load_ms"],
                                       "parameter_bytes": result["parameter_bytes"]}), flush=True)
    base = text.tokenizer.encode(PROMPT, add_special_tokens=False)
    if not base:
        raise RuntimeError("Benchmark prompt tokenized to nothing")
    result["prompt_base_token_ids"] = base
    def input_ids(length):
        values = (base * ((length + len(base) - 1) // len(base)))[:length]
        return torch.tensor([values], dtype=torch.long, device="cuda")
    try:
        with torch.inference_mode():
            for length in args.prompt_lengths:
                ids = input_ids(length)
                print("WARMUP " + str(length), flush=True)
                for _ in range(args.warmups):
                    checkpoint("before_warmup", prompt_tokens=length)
                    warm = text.engine.generate(ids, max_new_tokens=args.new_tokens, eos_token_id=None)
                    del warm
                torch.cuda.synchronize()
                checkpoint("after_warmup", prompt_tokens=length)
                case = {"prompt_tokens": length,
                        "input_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                        "samples": []}
                for repetition in range(args.repetitions):
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    checkpoint("before_sample", prompt_tokens=length, repetition=repetition)
                    started = time.perf_counter()
                    generated = text.engine.generate(ids, max_new_tokens=args.new_tokens, eos_token_id=None)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    assert generated.shape[1] == length + args.new_tokens
                    assert torch.equal(generated[:, :length], ids)
                    case["samples"].append({"repetition": repetition, "wall_ms": elapsed,
                        "generated_tokens_per_second": args.new_tokens * 1000 / elapsed,
                        "output_sha256": output_digest(generated, length), "memory": memory(torch)})
                    del generated
                    checkpoint("after_sample", prompt_tokens=length, sample=case["samples"][-1])
                    print("SAMPLE " + json.dumps({"prompt_tokens": length, "repetition": repetition,
                                                 "wall_ms": elapsed}), flush=True)
                case["summary"] = summarize(case["samples"])
                checkpoint("before_events", prompt_tokens=length)
                case["event_diagnostic"] = timed_diagnostic(torch, text.engine, ids, args.new_tokens)
                case["event_diagnostic"]["output_matches_baseline"] = (
                    case["event_diagnostic"]["output_sha256"] == case["samples"][0]["output_sha256"])
                result["cases"].append(case)
                write_json(result_path, result)
                del ids
            if args.profile:
                checkpoint("before_profile")
                result["status"] = "profiling"
                write_json(result_path, result)
                print("PROFILING", flush=True)
                result["profile"] = profile_request(torch, text.engine, input_ids(args.profile_length),
                                                     args.profile_new_tokens, args.output_dir)
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["finished"] = datetime.now(timezone.utc).isoformat()
        result["final_memory"] = memory(torch)
        write_json(result_path, result)
        if result["cases"]:
            with (args.output_dir / "samples.csv").open("w", newline="") as stream:
                fields = ["prompt_tokens", "repetition", "wall_ms", "generated_tokens_per_second",
                          "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "output_sha256"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for case in result["cases"]:
                    for row in case["samples"]:
                        writer.writerow({key: case["prompt_tokens"] if key == "prompt_tokens" else
                                         row["memory"][key] if key in row["memory"] else row[key] for key in fields})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=positive, nargs="+", default=[32, 256, 1024])
    parser.add_argument("--new-tokens", type=positive, default=128)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--warmups", type=positive, default=1)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--cuda-limit-gib", type=positive,
                        help="Optional PyTorch CUDA allocator cap; not a whole-system memory limit")
    parser.add_argument("--min-available-gib", type=positive, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-length", type=positive, default=256)
    parser.add_argument("--profile-new-tokens", type=positive, default=8)
    args = parser.parse_args(argv)
    # Refusing an existing directory must never rewrite a previous run's result.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        path = args.output_dir / "results.json"
        if path.exists():
            result = json.loads(path.read_text())
            result.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                          finished=datetime.now(timezone.utc).isoformat())
            write_json(path, result)
        raise


if __name__ == "__main__":
    main()
