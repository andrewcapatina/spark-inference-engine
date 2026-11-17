"""Checkpoint scalar/header audit only; never load a whole model."""
from pathlib import Path
import sys,json
import torch
from safetensors import safe_open
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True)
root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];raw=json.loads((root/'config.json').read_text());cfg=raw.get('text_config',raw)
def scalar(name):
 with safe_open(str(root/index[name]),framework='pt',device='cpu') as f:t=f.get_tensor(name)
 assert t.numel()==1
 return {'value':float(t.float().item()),'dtype':str(t.dtype),'raw_bytes':t.reshape(-1).view(torch.uint8).tolist()}
rows=[]
for i in range(cfg['num_hidden_layers']):
 prefix=f'model.language_model.layers.{i}.mlp.';gate=prefix+'gate_proj.';up=prefix+'up_proj.'
 if gate+'weight_packed' in index:
  a,b=scalar(gate+'input_global_scale'),scalar(up+'input_global_scale');ga,gb=scalar(gate+'weight_global_scale'),scalar(up+'weight_global_scale');rows.append({'layer':i,'backend':'nvfp4','gate_input_scale':a,'up_input_scale':b,'input_scale_bits_equal':a==b,'gate_weight_global_scale':ga,'up_weight_global_scale':gb})
 else:
  assert gate+'weight_scale' in index and up+'weight_scale' in index
  rows.append({'layer':i,'backend':'fp8','input_preparation':'Both FP8Linear use same BF16 row absmax/float scaling/division/cast independent of weight scales'})
d={'status':'complete','mode':mode,'scope':'CPU checkpoint headers/scalar scales only; no model loaded or generation performed','hidden_size':cfg['hidden_size'],'intermediate_size':cfg['intermediate_size'],'layers':rows,'nvfp4_pairs':sum(r['backend']=='nvfp4' for r in rows),'nvfp4_equal_input_scales':sum(r.get('input_scale_bits_equal',False) for r in rows),'fp8_pairs':sum(r['backend']=='fp8' for r in rows),'baseline_expression':'down_proj(silu(gate_proj(x)) * up_proj(x))'};(out/'results.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='layers'}),flush=True)
