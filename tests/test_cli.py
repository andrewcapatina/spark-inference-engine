"""CLI behavior without loading model weights; real CLI smoke runs are separate."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("native_cli", Path(__file__).parents[1] / "spark_engine/cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.config("qwen3_5")
        self.text = Mock()
        self.text.generate_text.return_value = " native continuation"
        self.text.chat.side_effect = ["First reply", "Second reply", "Third reply"]

    def config(self, kind):
        (self.model / "config.json").write_text(json.dumps({"model_type": kind}))

    def run_cli(self, args=(), stdin="", text=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch.object(cli.sys, "stdin", io.StringIO(stdin)):
            with patch.object(cli, "_load_qwen", return_value=text or self.text) as load:
                code = cli.main(["--model", str(self.model), *args])
        return code, out.getvalue(), err.getvalue(), load

    def test_raw_completion_uses_existing_adapter(self):
        code, out, err, load = self.run_cli(["--prompt", "Hello"])
        self.assertEqual(code, 0)
        self.assertEqual(out, " native continuation\n")
        self.assertIn("Loading", err)
        load.assert_called_once_with(self.model, "qwen", "cuda")
        self.text.generate_text.assert_called_once_with("Hello", max_new_tokens=4, temperature=0)

    def test_interactive_history_and_single_load(self):
        code, out, _, load = self.run_cli(stdin="Hello\nWhat did I say?\n/exit\n")
        self.assertEqual(code, 0)
        load.assert_called_once()
        calls = self.text.chat.call_args_list
        self.assertEqual(calls[1].args[0], [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "First reply"},
            {"role": "user", "content": "What did I say?"}])
        self.assertEqual(calls[0].kwargs, {"max_new_tokens": 128, "temperature": 0})
        self.assertIn("Assistant: Second reply", out)

    def test_reset_preserves_system_message(self):
        code, _, err, _ = self.run_cli(["--system", "Be brief."], "Hi\n/reset\nAgain\nquit\n")
        self.assertEqual(code, 0)
        self.assertIn("Conversation reset.", err)
        self.assertEqual(self.text.chat.call_args_list[1].args[0], [
            {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Again"}])

    def test_cancelled_turn_is_not_added_to_history(self):
        self.text.chat.side_effect = ["Kept", KeyboardInterrupt(), "Recovered"]
        code, out, err, _ = self.run_cli(stdin="First\nCancel this\nNext\n/exit\n")
        self.assertEqual(code, 0)
        self.assertIn("unfinished turn discarded", err)
        self.assertIn("Assistant: Recovered", out)
        self.assertEqual(self.text.chat.call_args_list[2].args[0], [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Kept"}, {"role": "user", "content": "Next"}])

    def test_chat_accepts_first_prompt_and_token_limit(self):
        code, _, _, _ = self.run_cli(["--chat", "--prompt", "Start here", "--max-new-tokens", "8"])
        self.assertEqual(code, 0)
        self.text.chat.assert_called_once_with([{"role": "user", "content": "Start here"}],
                                               max_new_tokens=8, temperature=0)
        self.text.generate_text.assert_not_called()

    def test_eof_and_commands_do_not_generate(self):
        for stdin in ("", "\n/help\n/quit\n", "exit\n"):
            with self.subTest(stdin=stdin):
                self.assertEqual(self.run_cli(stdin=stdin)[0], 0)
        self.text.chat.assert_not_called()

    def test_ctrl_c_at_input_exits(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_cli()[0], 0)

    def test_ctrl_c_during_load_returns_130(self):
        with redirect_stderr(io.StringIO()), patch.object(cli, "_load_qwen", side_effect=KeyboardInterrupt):
            self.assertEqual(cli.main(["--model", str(self.model)]), 130)

    def test_runtime_error_is_readable(self):
        self.text.chat.side_effect = RuntimeError("test allocation failure")
        code, _, err, _ = self.run_cli(stdin="Hello\n")
        self.assertEqual(code, 1)
        self.assertIn("Error: test allocation failure", err)

    def test_gguf_routes_to_qwen_loader(self):
        self.model = self.model / "model.gguf"
        self.model.touch()
        code, _, _, load = self.run_cli(["--prompt", "Hello"])
        self.assertEqual(code, 0)
        load.assert_called_once_with(self.model, "qwen-gguf", "cuda")

    def test_unsupported_architecture_fails_before_loading(self):
        self.config("qwen3_moe")
        code, _, err, load = self.run_cli(["--prompt", "Hi"])
        self.assertEqual(code, 1)
        self.assertIn("Unsupported model_type", err)
        load.assert_not_called()

    def test_explicit_architecture_and_nested_config(self):
        self.config("custom")
        self.assertEqual(self.run_cli(["--architecture", "qwen", "--prompt", "Hi"])[0], 0)
        (self.model / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3_5_text"}}))
        self.assertEqual(cli._model_kind(self.model, "auto"), "qwen")

    def test_invalid_options_fail_before_loading(self):
        for args in (["--max-new-tokens", "0"], ["--max-new-tokens", "-1"],
                     ["--device", "cpu"], ["--prompt", "Hi", "--system", "Hello"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.run_cli(args)
            self.assertEqual(error.exception.code, 2)

    def test_missing_model_path_is_readable(self):
        self.model = self.model / "missing"
        code, _, err, load = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)
        load.assert_not_called()

    def test_legacy_llama_completion(self):
        self.config("llama")
        with patch.object(cli, "_llama_completion", return_value="Hello native") as generate:
            code, out, _, load = self.run_cli(["--prompt", "Hello"])
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hello native\n")
        generate.assert_called_once_with(self.model, "Hello", "cuda", 4)
        load.assert_not_called()

    def test_llama_chat_is_not_silently_emulated(self):
        self.config("llama")
        with self.assertRaises(SystemExit) as error:
            self.run_cli()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
