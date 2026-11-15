"""Sequential tiny-model golden fixture; references expand tiny weights only in tests."""
import sys,json,tempfile,gc,torch
from pathlib import Path
from safetensors.torch import save_file
sys.path.insert(0,'/workspace')
from spark_engine.qwen_model import QwenCausalLM
from spark_engine.qwen_quantized import load_quantized_qwen
import flashinfer

def unpack(p,s,g):
 lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device=p.device)
 return torch.stack((lut[(p&15).long()],lut[(p>>4).long()]),-1).flatten(-2)*s.float().repeat_interleave(16,-1)/g
class FixtureFP4(torch.nn.Module):
 def __init__(self,p,s,wg,ig):
  super().__init__();self.register_buffer('dense',unpack(p,s,wg));self.register_buffer('ig',ig)
 def forward(self,x):
  q,s=flashinfer.fp4_quantize(x.reshape(-1,x.shape[-1]).contiguous(),self.ig,is_sf_swizzled_layout=False);s=s.view(torch.float8_e4m3fn).reshape(q.shape[0],x.shape[-1]//16)
  return (unpack(q.view(torch.uint8),s,self.ig)@self.dense.T).to(x.dtype).reshape(*x.shape[:-1],self.dense.shape[0])
class FixtureFP8(torch.nn.Module):
 def __init__(self,w,s):super().__init__();self.register_buffer('dense',w.float()*s.float())
 def forward(self,x):
  z=x.reshape(-1,x.shape[-1]).float();s=z.abs().amax(-1,keepdim=True).div(448).clamp_min(1e-12);q=(z/s).to(torch.float8_e4m3fn).float()*s
  return (q@self.dense.T).to(x.dtype).reshape(*x.shape[:-1],self.dense.shape[0])
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 return x

def create_fixture(root):
 torch.manual_seed(77)
 cfg={'vocab_size':256,'hidden_size':128,'intermediate_size':256,'num_hidden_layers':2,'num_attention_heads':1,'num_key_value_heads':1,'head_dim':128,'linear_num_key_heads':1,'linear_num_value_heads':1,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4,'rms_norm_eps':1e-6,'layer_types':['full_attention','linear_attention'],'rope_parameters':{'rope_theta':10000000.,'partial_rotary_factor':.25}}
 teacher=QwenCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval();source={('lm_head.weight' if k=='lm_head.weight' else 'model.language_model.'+k):cpu(v).contiguous() for k,v in teacher.state_dict().items()}
 def replace(name,module):parent,leaf=name.rsplit('.',1) if '.' in name else ('',name);setattr(teacher.get_submodule(parent),leaf,module)
 for name,mod in list(teacher.named_modules()):
  if not isinstance(mod,torch.nn.Linear) or name.endswith(('in_proj_a','in_proj_b')):continue
  prefix=name if name=='lm_head' else 'model.language_model.'+name;weight=mod.weight.detach();source.pop(prefix+'.weight')
  if '.mlp.' in name:
   wg=torch.tensor([2.75],device='cuda');ig=torch.tensor([3.25],device='cuda');packed,scale=flashinfer.fp4_quantize(weight,wg,is_sf_swizzled_layout=False);packed=packed.view(torch.uint8);scale=scale.view(torch.float8_e4m3fn).reshape(weight.shape[0],weight.shape[1]//16)
   source.update({prefix+'.weight_packed':cpu(packed),prefix+'.weight_scale':cpu(scale),prefix+'.weight_global_scale':cpu(wg),prefix+'.input_global_scale':cpu(ig)});replace(name,FixtureFP4(packed,scale,wg,ig))
  else:
   scale=(weight.float().abs().amax(-1,keepdim=True)/448).clamp_min(1e-6).to(torch.bfloat16);q=(weight.float()/scale.float()).to(torch.float8_e4m3fn);source.update({prefix+'.weight':cpu(q),prefix+'.weight_scale':cpu(scale)});replace(name,FixtureFP8(q,scale))
 save_file(source,str(root/'model.safetensors'));(root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in source}}));(root/'config.json').write_text(json.dumps({'text_config':cfg,'quantization_config':{'quant_method':'compressed-tensors','format':'mixed-precision'}}))
 ids=torch.tensor([[1,2,3,4,5]],device='cuda');logits,cache=teacher.prefill(ids);decoded,_=teacher.decode(torch.tensor([[6]],device='cuda'),cache)
 golden={'forward':cpu(teacher(ids)),'prefill':cpu(logits),'decode':cpu(decoded),'generated':cpu(teacher.generate(ids,4,None))}
 del teacher;return golden

def run():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);golden=create_fixture(root);gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
  # The fixture teacher is gone before the candidate model is materialized.
  model=load_quantized_qwen(root);assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters());assert all(b.device.type=='cuda' for b in model.buffers());assert model.quantization_metadata['nvfp4_linears']==6 and model.quantization_metadata['fp8_linears']==8;assert model.quantization_metadata['kv_cache_dtype']=='bfloat16'
  ids=torch.tensor([[1,2,3,4,5]],device='cuda');logits,cache=model.prefill(ids);decoded,_=model.decode(torch.tensor([[6]],device='cuda'),cache)
  outputs={'forward':model(ids),'prefill':logits,'decode':decoded}
  for name,actual in outputs.items():torch.testing.assert_close(actual.cpu(),golden[name],rtol=.03,atol=.03125)
  generated=model.generate(ids,4,None);assert torch.equal(generated.cpu(),golden['generated'])
  assert any(b.dtype==torch.uint8 for b in model.buffers()) and any(b.dtype==torch.float8_e4m3fn for b in model.buffers())
  result={'status':'tiny_quantized_loader_passed','scope':'Synthetic2layer hybrid checkpoint, sequential independent-reference model and native loader; not actual27B validation','generation_ids':generated.cpu().tolist(),'metadata':model.quantization_metadata,'max_abs_errors':{k:float((outputs[k].cpu()-golden[k]).abs().max()) for k in outputs}}
  del model,outputs,cache,logits,decoded,generated;gc.collect();torch.cuda.empty_cache()
  index=root/'model.safetensors.index.json';original=json.loads(index.read_text());bad=json.loads(index.read_text());bad['weight_map'].pop('model.language_model.norm.weight');index.write_text(json.dumps(bad))
  try:load_quantized_qwen(root)
  except (KeyError,ValueError,RuntimeError,AssertionError):result['missing_weight_rejected']=True
  else:raise AssertionError('Missing required norm accepted')
  bad=json.loads(json.dumps(original));bad['weight_map']['model.language_model.unknown.weight']='model.safetensors';index.write_text(json.dumps(bad))
  try:load_quantized_qwen(root)
  except (KeyError,ValueError,RuntimeError,AssertionError):result['unexpected_weight_rejected']=True
  else:raise AssertionError('Unexpected text tensor accepted')
  print(json.dumps(result),flush=True)
if __name__=='__main__':
 with torch.inference_mode():run()
