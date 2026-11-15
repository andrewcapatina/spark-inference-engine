"""CPU tokenizer API compatibility fixture; dummy generation is not an inference test."""
import sys,json,torch
sys.path.insert(0,'/workspace')
from transformers import PreTrainedTokenizerFast
from spark_engine.qwen_text import NativeQwenText
class Dummy(torch.nn.Module):
 def __init__(self,tokenizer):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.tokenizer=tokenizer
 def generate(self,ids,max_new_tokens,eos_token_id):
  assert isinstance(ids,torch.Tensor) and ids.ndim==2 and ids.device.type=='cpu'
  suffix=torch.tensor([self.tokenizer.encode('4',add_special_tokens=False)])
  return torch.cat((ids,suffix),dim=1)
if __name__=='__main__':
 t=PreTrainedTokenizerFast.from_pretrained(sys.argv[1],local_files_only=True);text=NativeQwenText(Dummy(t),t);answer=text.chat([{'role':'user','content':'What is 2 + 2?'}]);assert answer=='4' and not torch.cuda.is_initialized();print(json.dumps({'status':'cpu_chat_tokenizer_api_passed','model_directory':sys.argv[1],'scope':'Tokenizer/dummy-wrapper compatibility only; no CUDA or real model loaded'}))
