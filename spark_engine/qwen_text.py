import torch


class NativeQwenText:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    def generate_text(self, prompt, max_new_tokens=32, temperature=0):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.engine.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        with torch.inference_mode():
            out = self.engine.generate(input_ids, max_new_tokens=max_new_tokens)
        new_ids = out[:, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
