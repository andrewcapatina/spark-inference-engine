"""Independent component references; never a claim of full-checkpoint support."""
import sys,json,torch
from pathlib import Path
sys.path.insert(0,'/workspace')
from spark_engine.qwen_nvfp4 import NVFP4Linear,swizzle_nvfp4_scales
import flashinfer

def unpack(packed,scales,global_inverse):
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device=packed.device)
    values=torch.stack((lut[(packed&15).long()],lut[(packed>>4).long()]),dim=-1).flatten(-2)
    return values*scales.float().repeat_interleave(16,-1)/global_inverse.float()

def run():
    torch.manual_seed(42);checked=[]
    for m,n,k in [(1,128,128),(3,128,256),(16,256,128),(128,256,256)]:
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        ig=torch.tensor([3.25],device='cuda');wg=torch.tensor([2.75],device='cuda')
        wp,ws=flashinfer.fp4_quantize(w,wg,is_sf_swizzled_layout=False);ws=ws.view(torch.float8_e4m3fn).reshape(n,k//16)
        xp,xs=flashinfer.fp4_quantize(x,ig,is_sf_swizzled_layout=False);xs=xs.view(torch.float8_e4m3fn).reshape(m,k//16)
        layer=NVFP4Linear(wp.view(torch.uint8),ws,wg,ig)
        assert any(b.dtype==torch.uint8 and b.numel()==n*k//2 for b in layer.buffers())
        for shape in ((m,k),(1,m,k)):
            actual=layer(x.reshape(shape));expected=(unpack(xp.view(torch.uint8),xs,ig)@unpack(wp.view(torch.uint8),ws,wg).T).to(torch.bfloat16).reshape(*shape[:-1],n)
            torch.testing.assert_close(actual,expected,rtol=.01,atol=.03125)
        assert sum(b.numel()*b.element_size() for b in layer.buffers())<n*k,'Weights inflated to BF16'
        checked.append({'M':m,'N':n,'K':k,'max_abs_error':float((actual.float()-expected.float()).abs().max())})
    # Independent address map verifies interleave, including padding.
    s=torch.arange(130*7,dtype=torch.float32).reshape(130,7).remainder(10).to(torch.float8_e4m3fn)
    got=swizzle_nvfp4_scales(s).view(torch.uint8).flatten();raw=s.view(torch.uint8)
    assert got.numel()==256*8
    for row in (0,1,31,32,63,127,128,129):
        for col in range(7):
            offset=(((((row//128)*2+col//4)*32+row%32)*4+(row%128)//32)*4+col%4)
            assert got[offset]==raw[row,col]
    for bad in (torch.zeros(1,128,device='cuda'),torch.zeros(1,127,device='cuda',dtype=torch.bfloat16)):
        try:layer(bad)
        except (ValueError,TypeError,RuntimeError,AssertionError):pass
        else:raise AssertionError('Invalid input accepted')
    print(json.dumps({'status':'tiny_nvfp4_operators_passed','cases':checked,'scope':'independent quantized matmul and scale layout only; no full model validation'}),flush=True)
if __name__=='__main__':
    with torch.inference_mode():run()
