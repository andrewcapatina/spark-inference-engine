"""Native greedy output and call-budget contract; no checkpoint required."""
from types import SimpleNamespace
import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required; run in the experiment container')
class GenerationBudgetTests(unittest.TestCase):
    def test_token_limit_and_eos_preserve_outputs_without_unused_decode(self):
        from spark_engine.qwen_model import QwenCausalLM
        cases = [(0, None, [2]), (1, None, [2]),
                 (4, None, [2, 3, 4, 5]), (4, 2, [2]),
                 (4, 3, [2, 3]), (4, 5, [2, 3, 4, 5])]
        for budget, eos, sequence in cases:
            with self.subTest(budget=budget, eos=eos):
                calls = []
                def scores(index):
                    logits = torch.full((1, 1, 16), -1000.)
                    logits[0, 0, sequence[index]] = 1.
                    return logits
                def prefill(ids):
                    calls.append('prefill')
                    return scores(0), []
                def decode(ids, cache):
                    calls.append('decode')
                    return scores(min(calls.count('decode'), len(sequence) - 1)), []
                engine = SimpleNamespace(prefill=prefill, decode=decode)
                prompt = torch.tensor([[7, 8]], dtype=torch.long)
                with torch.inference_mode():
                    actual = QwenCausalLM.generate(engine, prompt, budget, eos)
                expected = sequence[:budget]
                if eos in expected:
                    expected = expected[:expected.index(eos) + 1]
                self.assertEqual(actual.tolist(), [[7, 8] + expected])
                self.assertEqual(actual.dtype, prompt.dtype)
                self.assertEqual(actual.device, prompt.device)
                self.assertEqual(calls, ['prefill'] + ['decode'] * max(0, len(expected) - 1))


if __name__ == '__main__':
    unittest.main()
