import unittest
import torch
from torch import nn
from stylelib.adapter import install_style,style_modules,style_parameters,style_state,load_style_state

class Project(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(8,16)
    def forward(self,x):return torch.tanh(self.proj(x))

class Expanded(nn.Module):
    def __init__(self):
        super().__init__();self.transformer_blocks=nn.ModuleList()
        for _ in range(2):
            b=nn.Module();b.attn=nn.Module();b.attn.orig=nn.Module()
            for n in ('to_q','to_k','to_v'):setattr(b.attn.orig,n,nn.Linear(8,8))
            b.attn.orig.to_out=nn.ModuleList([nn.Linear(8,8)])
            b.ff=nn.Module();b.ff.net=nn.Sequential(Project(),nn.Identity(),nn.Linear(16,8))
            self.transformer_blocks.append(b)
    def forward(self,x):
        for b in self.transformer_blocks:
            a=b.attn.orig;x=x+a.to_out[0](torch.tanh(a.to_q(x)+a.to_k(x)+a.to_v(x)));x=x+b.ff.net(x)
        return x

class MLPScope(unittest.TestCase):
    def test_freeze_gradients_and_restore_mlp(self):
        torch.manual_seed(23);m=Expanded();x=torch.randn(4,8);before=m(x).detach()
        install_style(m,rank=4,alpha=4,expected_modules=12,expected_parameters=896)
        torch.testing.assert_close(m(x),before,rtol=0,atol=0)
        opt=torch.optim.AdamW(style_parameters(m),lr=.01)
        for _ in range(2):
            opt.zero_grad();m(x).square().mean().backward()
            self.assertTrue(all(a.style_B.grad is not None and a.style_B.grad.abs().sum()>0 for _,a in style_modules(m)))
            opt.step()
        self.assertTrue(all(a.style_A.grad is not None and a.style_A.grad.abs().sum()>0 for _,a in style_modules(m)))
        other=Expanded();install_style(other,rank=4,alpha=4,expected_modules=12,expected_parameters=896)
        state=style_state(m);load_style_state(other,state)
        self.assertTrue(all(torch.equal(state[k],style_state(other)[k]) for k in state))
        self.assertTrue(all(not p.requires_grad for n,p in m.named_parameters() if not n.endswith(('.style_A','.style_B'))))
