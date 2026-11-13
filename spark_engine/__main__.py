import argparse
import os

import torch

from .model import CausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = CausalLM.from_pretrained(args.model).to(args.device).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        out = model.generate(inputs["input_ids"], max_new_tokens=args.max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
