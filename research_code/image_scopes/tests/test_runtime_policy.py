import unittest
import torch
from torch import nn
from stylelib.model import set_inner_checkpoints
from src.utils.activation_checkpointing import checkpoint_forward

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.attn=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,8));self.ff=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,8))
    def forward(self,x):
        x=x+self.attn(x);return x+self.ff(x)
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.transformer_blocks=nn.ModuleList([Block(),Block()])
        for b in self.transformer_blocks:
            checkpoint_forward(b.attn);checkpoint_forward(b.ff);checkpoint_forward(b)
    def forward(self,x):
        for b in self.transformer_blocks:x=b(x)
        return x
class RuntimePolicy(unittest.TestCase):
    def test_toggle_preserves_outputs_gradients_and_names(self):
        torch.manual_seed(13);m=Model();x=torch.randn(5,8);names=list(dict(m.named_parameters()))
        def run(enabled):
            set_inner_checkpoints(m,enabled);m.zero_grad();y=m(x);y.square().mean().backward()
            return y.detach(),{n:p.grad.clone() for n,p in m.named_parameters()}
        baseline,grad=run(True)
        for enabled in (False,True,False):
            output,new=run(enabled);torch.testing.assert_close(output,baseline,rtol=0,atol=0)
            self.assertEqual(names,list(dict(m.named_parameters())))
            for n in grad:torch.testing.assert_close(new[n],grad[n],rtol=0,atol=0)
