"""Independent FP8 component checks, not model-generation acceptance."""
import sys,json,torch
sys.path.insert(0,'/workspace')
from spark_engine.qwen_fp8 import FP8Linear

def run():
 torch.manual_seed(43);cases=[]
 for m,n,k in ((1,128,128),(3,128,256),(16,256,128),(128,256,256)):
  weight=(torch.randn(n,k,device='cuda')*16).to(torch.float8_e4m3fn);scale=torch.linspace(.01,.1,n,device='cuda').to(torch.bfloat16).reshape(n,1);layer=FP8Linear(weight,scale)
  assert any(b.dtype==torch.float8_e4m3fn and b.numel()==n*k for b in layer.buffers())
  assert sum(b.numel()*b.element_size() for b in layer.buffers())<2*n*k
  x=torch.randn(k,m,device='cuda',dtype=torch.bfloat16).T
  a_scale=x.float().abs().amax(-1,keepdim=True).clamp_min(1e-6)/448
  aq=(x.float()/a_scale).clamp(-448,448).to(torch.float8_e4m3fn)
  ref=((aq.float()*a_scale)@(weight.float()*scale.float()).T).to(torch.bfloat16)
  for shape in ((m,k),(1,m,k)):
   got=layer(x.reshape(shape));torch.testing.assert_close(got,ref.reshape(*shape[:-1],n),rtol=.01,atol=.03125)
  assert torch.equal(layer(torch.zeros_like(x)),torch.zeros(m,n,device='cuda',dtype=torch.bfloat16))
  cases.append({'M':m,'N':n,'K':k,'max_abs_error':float((got.float()-ref.reshape_as(got).float()).abs().max())})
 print(json.dumps({'status':'tiny_fp8_operators_passed','cases':cases,'scope':'Independent scaled FP8 matmul only; no full checkpoint validation'}),flush=True)
if __name__=='__main__':
 with torch.inference_mode():run()
