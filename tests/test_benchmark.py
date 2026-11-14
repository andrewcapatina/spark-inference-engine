"""Measurement accounting and observer isolation; no model weights required."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("qwen_benchmark", Path(__file__).parents[1] / "tools/benchmark_qwen.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Event:
    def __init__(self, milliseconds):
        self.milliseconds = milliseconds

    def elapsed_time(self, other):
        return other.milliseconds - self.milliseconds


class MeasurementTests(unittest.TestCase):
    def test_decode_throughput_counts_intervals_after_first_token(self):
        result = bench.event_metrics(Event(0), Event(24), [(Event(0), Event(10))],
            [(Event(11), Event(15)), (Event(16), Event(20)), (Event(21), Event(24))],
            [Event(10), Event(15), Event(20)])
        self.assertEqual(result["first_token_cuda_ms"], 10)
        self.assertEqual(result["prefill_cuda_ms"], 10)
        self.assertEqual(result["decode_tokens_per_second_cuda"], 200)
        self.assertEqual(result["observed_decode_calls"], 3)
        self.assertEqual(result["final_unused_decode_cuda_ms"], 3)
        self.assertEqual(result["total_cuda_ms"], 24)

    def test_single_generated_token_has_no_decode_token_window(self):
        result = bench.event_metrics(Event(0), Event(15), [(Event(0), Event(10))],
            [(Event(11), Event(15))], [Event(10)])
        self.assertIsNone(result["decode_tokens_per_second_cuda"])
        self.assertIsNone(result["decode_token_window_cuda_ms"])

    def test_observation_preserves_returns_and_restores_class_methods(self):
        class Engine:
            def prefill(self, value):
                return value
            def decode(self, value):
                return value
        engine = Engine()
        before = dict(vars(engine))
        calls = []
        sentinel = object()
        def factory(name, original):
            def wrapped(*args, **kwargs):
                calls.append(name)
                return original(*args, **kwargs)
            return wrapped
        with bench.observe(engine, factory):
            self.assertIs(engine.prefill(sentinel), sentinel)
            self.assertIs(engine.decode(sentinel), sentinel)
        self.assertEqual(calls, ["prefill", "decode"])
        self.assertEqual(vars(engine), before)

    def test_observer_restores_existing_instance_override_on_error(self):
        first, second = lambda value: value, lambda value: value
        engine = SimpleNamespace(prefill=first, decode=second)
        with self.assertRaisesRegex(RuntimeError, "cancel"):
            with bench.observe(engine, lambda name, original: lambda value: None):
                raise RuntimeError("cancel")
        self.assertIs(engine.prefill, first)
        self.assertIs(engine.decode, second)

    def test_summary_uses_repetitions_and_preserves_output_difference(self):
        rows = [{"wall_ms": wall, "generated_tokens_per_second": rate,
                 "output_sha256": digest, "memory": {"cuda_peak_allocated_bytes": peak,
                 "cuda_peak_reserved_bytes": 100}} for wall, rate, digest, peak in
                [(30, 1, "same", 30), (10, 3, "same", 10), (20, 2, "different", 20)]]
        result = bench.summarize(rows)
        self.assertEqual(result["median_wall_ms"], 20)
        self.assertEqual(result["median_generated_tokens_per_second"], 2)
        self.assertEqual(result["peak_cuda_allocated_bytes"], 30)
        self.assertFalse(result["repeated_outputs_match"])

    def test_atomic_json_is_complete_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            bench.write_json(path, {"status": "loading"})
            bench.write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_existing_run_is_preserved_when_output_directory_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            bench.write_json(path, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                bench.main(["--model", "unused", "--output-dir", directory])
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
