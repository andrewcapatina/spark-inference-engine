"""Exercise the existing public API on the independent tiny quantized fixture."""
import torch
import test_quantized_loader as fixture
fixture.load_quantized_qwen=fixture.QwenCausalLM.from_pretrained
if __name__=='__main__':
 with torch.inference_mode():fixture.run()
