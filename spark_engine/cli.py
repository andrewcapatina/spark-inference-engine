"""Command-line presentation for the existing native inference APIs."""
import argparse
import json
from pathlib import Path
import sys


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser():
    parser = argparse.ArgumentParser(description="Native Llama completions and Qwen chat.")
    parser.add_argument("--model", required=True, type=Path,
                        help="Local model directory or Qwen3.5 GGUF file")
    parser.add_argument("--architecture", choices=("auto", "llama", "qwen"), default="auto",
                        help="Model architecture (default: detect from config.json/file format)")
    parser.add_argument("--prompt", help="One-shot raw completion, or the first question with --chat")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive Qwen chat (also the default when --prompt is omitted)")
    parser.add_argument("--system", help="System message for Qwen chat")
    parser.add_argument("--max-new-tokens", type=_positive_int,
                        help="Reply limit (default: 128 for chat, 4 for raw completion)")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    return parser


def _model_kind(path, architecture):
    if not path.exists():
        raise ValueError(f"Model path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError("Expected a model directory or a .gguf file")
        if architecture == "llama":
            raise ValueError("GGUF loading is currently supported for Qwen3.5 only")
        return "qwen-gguf"
    if architecture != "auto":
        return architecture
    config = json.loads((path / "config.json").read_text())
    kind = config.get("model_type") or config.get("text_config", {}).get("model_type")
    if kind in ("qwen3_5", "qwen3_5_text"):
        return "qwen"
    if kind == "llama":
        return "llama"
    raise ValueError(f"Unsupported model_type {kind!r}; select --architecture for a compatible local model")


def _load_qwen(path, kind, device):
    import torch

    if kind == "qwen-gguf":
        from .qwen_gguf import load_qwen_gguf, read_metadata
        architecture = read_metadata(str(path)).get("general.architecture")
        if architecture != "qwen35":
            raise ValueError(f"Unsupported GGUF architecture: {architecture!r}; expected qwen35")
        return load_qwen_gguf(str(path), device=device, dtype=torch.bfloat16)
    from .qwen_text import NativeQwenText
    return NativeQwenText.from_pretrained(str(path), device=device, dtype=torch.bfloat16)


def _llama_completion(path, prompt, device, max_new_tokens):
    # Preserve the original CLI's offline loading and prompt-inclusive output.
    import torch
    from transformers import AutoTokenizer
    from .model import CausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = CausalLM.from_pretrained(str(path)).to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _chat(text, max_new_tokens, system=None, first_prompt=None):
    initial = [{"role": "system", "content": system}] if system is not None else []
    history = list(initial)
    pending = first_prompt
    print("Qwen chat. /reset clears history; /exit quits. Ctrl-C cancels a reply.", file=sys.stderr)
    while True:
        if pending is None:
            try:
                prompt = input("You: " if sys.stdin.isatty() else "")
            except (EOFError, KeyboardInterrupt):
                if sys.stdin.isatty():
                    print()
                return 0
        else:
            prompt, pending = pending, None
        if not prompt.strip():
            continue
        command = prompt.strip()
        if command in ("/exit", "/quit", "exit", "quit"):
            return 0
        if command == "/reset":
            history = list(initial)
            print("Conversation reset.", file=sys.stderr)
            continue
        if command == "/help":
            print("/reset: clear history; /exit or /quit: quit; Ctrl-C: cancel reply.", file=sys.stderr)
            continue
        messages = history + [{"role": "user", "content": prompt}]
        try:
            reply = text.chat(messages, max_new_tokens=max_new_tokens, temperature=0)
        except KeyboardInterrupt:
            print("\nReply cancelled; unfinished turn discarded.", file=sys.stderr)
            continue
        print(f"Assistant: {reply}", flush=True)
        history = messages + [{"role": "assistant", "content": reply}]


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    path = args.model.expanduser()
    chat = args.chat or args.prompt is None
    limit = args.max_new_tokens if args.max_new_tokens is not None else (128 if chat else 4)
    try:
        kind = _model_kind(path, args.architecture)
        if args.system is not None and not chat:
            parser.error("--system requires chat mode; add --chat")
        if kind == "llama":
            if chat:
                parser.error("Interactive chat currently supports Qwen; use --prompt for Llama completion")
            print(_llama_completion(path, args.prompt, args.device, limit))
            return 0
        if args.device.split(":", 1)[0] != "cuda":
            parser.error("Qwen CLI inference requires a CUDA device")
        print(f"Loading {path} ...", file=sys.stderr, flush=True)
        text = _load_qwen(path, kind, args.device)
        if chat:
            return _chat(text, limit, args.system, args.prompt)
        print(text.generate_text(args.prompt, max_new_tokens=limit, temperature=0))
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
