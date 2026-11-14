import torch
from transformers import PreTrainedTokenizerFast
from spark_engine.qwen_model import QwenCausalLM


class NativeQwenText:
    @classmethod
    def from_pretrained(cls, directory, dtype=torch.bfloat16, device="cuda"):
        engine = QwenCausalLM.from_pretrained(directory, dtype=dtype, device=device)
        engine.eval()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
        return cls(engine, tokenizer)
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    def generate_text(self, prompt, max_new_tokens=32, temperature=0):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.engine.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        with torch.inference_mode():
            out = self.engine.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_token_id)
        new_ids = out[:, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids[0], skip_special_tokens=True)

    def chat(self, messages, max_new_tokens=32, temperature=0):
        device = next(self.engine.parameters()).device
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False
        )
        input_ids = ids.to(device)
        with torch.inference_mode():
            out = self.engine.generate(
                input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id
            )
        new_ids = out[:, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
