"""Tiny genuine Hopper FP8 GEMM+backward gate. No real model or optimizer checkpoint."""
import json,torch
from torch import nn
from pathlib import Path
from stylelib.fp8_training import FP8FrozenLinear,FP8StyleLinear
from stylelib.adapter import StyleLinear

def main():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    base=nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
    q=FP8FrozenLinear(base);x=torch.randn(37,256,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.randn_like(x)
    ref=base(x);(ref*g).sum().backward();ref_grad=x.grad.clone();x.grad=None
    quant=q(x);(quant*g).sum().backward()
    forward_error=float((quant.float()-ref.float()).norm()/ref.float().norm());grad_error=float((x.grad.float()-ref_grad.float()).norm()/ref_grad.float().norm())
    assert forward_error<.08 and grad_error<.08 and torch.isfinite(x.grad).all()
    a=FP8StyleLinear(StyleLinear(base,rank=4,alpha=4));b=FP8StyleLinear(StyleLinear(nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16).requires_grad_(False),rank=4,alpha=4))
    model=nn.Sequential(a,nn.SiLU(),b);params=[a.style_A,a.style_B,b.style_A,b.style_B];opt=torch.optim.AdamW(params,lr=1e-3)
    for step in range(3):
        opt.zero_grad();model(x.detach()).float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert a.style_B.grad.norm()>0 and b.style_B.grad.norm()>0
        if step:assert a.style_A.grad.norm()>0 and b.style_A.grad.norm()>0
        opt.step()
    report={'GPU':torch.cuda.get_device_name(),'forward_relative_L2':forward_error,'input_gradient_relative_L2':grad_error,'three_actual_FP8_updates':True,'early_and_late_LoRA_gradients':True,'full_H3_tested':False,'LoRA_masters_FP32':all(p.dtype==torch.float32 for p in params),'passed':True}
    Path('fp8_cuda_gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
