"""CPU-only metadata/streaming contract checks; no GPU/model loading."""
import sys,tempfile,json,torch
from pathlib import Path
from safetensors.torch import save_file
sys.path.insert(0,'/workspace')
from spark_engine.qwen_quantized import quantized_checkpoint

def run():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);cfg={'text_config':{'hidden_size':128},'quantization_config':{'quant_method':'compressed-tensors','format':'mixed-precision'}};(root/'config.json').write_text(json.dumps(cfg))
  tensors={'model.language_model.norm.weight':torch.ones(128,dtype=torch.bfloat16),'model.language_model.layers.0.mlp.up_proj.weight_packed':torch.arange(16,dtype=torch.uint8).reshape(2,8),'lm_head.weight':torch.ones(16,16).to(torch.float8_e4m3fn)}
  save_file(tensors,str(root/'text.safetensors'));index={k:'text.safetensors' for k in tensors};index.update({'model.visual.fake':'absent-vision.safetensors','mtp.fake':'absent-mtp.safetensors','model.language_model.layers.0.self_attn.k_scale':'absent-kv.safetensors'});p=root/'model.safetensors.index.json'
  def write(m):p.write_text(json.dumps({'weight_map':m}))
  write(index)
  with quantized_checkpoint(root) as (c,get,keys):
   assert c['hidden_size']==128 and set(keys)=={'norm.weight','layers.0.mlp.up_proj.weight_packed','lm_head.weight'}
   for name,tensor in tensors.items():
    canonical=name.removeprefix('model.language_model.');actual=get(canonical);assert actual.device.type=='cpu' and actual.dtype==tensor.dtype;assert torch.equal(actual.view(torch.uint8),tensor.view(torch.uint8))
  checks=['canonical_mapping','excluded_shards_not_opened','dtype_bits_unchanged']
  try:
   with quantized_checkpoint(root) as (_,get,_):get('missing.weight')
  except KeyError:checks.append('original_exception_preserved')
  else:raise AssertionError('Original missing-tensor error not preserved')
  try:
   with quantized_checkpoint(root):pass
  except RuntimeError:checks.append('unconsumed_rejected')
  else:raise AssertionError('Unconsumed tensors accepted')
  for label,modified in [('path_escape',{**index,'lm_head.weight':'../outside.safetensors'}),('unknown_namespace',{**index,'unknown.weight':'text.safetensors'}),('duplicate',{**index,'model.language_model.lm_head.weight':'text.safetensors'})]:
   write(modified)
   try:
    with quantized_checkpoint(root):pass
   except (ValueError,RuntimeError,AssertionError):checks.append(label)
   else:raise AssertionError(label+' accepted')
  write({**index,'model.language_model.layers.0.self_attn.k_scale_extra':'text.safetensors'})
  try:
   with quantized_checkpoint(root) as (_,get,keys):
    assert 'layers.0.self_attn.k_scale_extra' in keys
    for k in keys-{'layers.0.self_attn.k_scale_extra'}:get(k)
  except RuntimeError:checks.append('exact_kv_calibration_filter')
  else:raise AssertionError('Unexpected calibration-like key silently ignored')
  print(json.dumps({'status':'cpu_quant_reader_passed','checks':checks,'scope':'CPU metadata only; no engine inference'}),flush=True)
if __name__=='__main__':run()
