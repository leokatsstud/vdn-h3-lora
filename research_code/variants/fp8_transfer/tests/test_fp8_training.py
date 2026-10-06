import unittest,torch
from torch import nn
from stylelib.fp8_training import FP8FrozenLinear,FP8StyleLinear,quantize_rows
from stylelib.adapter import StyleLinear

class FP8Training(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def base(self):return nn.Linear(64,64,bias=True).requires_grad_(False)
    def test_forward_matches_quantized_reference(self):
        b=self.base();q=FP8FrozenLinear(b,reference=True);x=torch.randn(19,64)
        a,s=quantize_rows(x);expected=(a.float()*s)@(q.weight_fp8.float()*q.weight_scale).T+b.bias
        torch.testing.assert_close(q(x),expected)
    def test_backward_accounts_for_nonuniform_weight_row_scales(self):
        b=self.base()
        with torch.no_grad():b.weight.mul_(torch.linspace(.01,20,64)[:,None])
        q=FP8FrozenLinear(b,reference=True);x=torch.randn(19,64,requires_grad=True);g=torch.randn(19,64)
        (q(x)*g).sum().backward()
        weighted=g*q.weight_scale.T;g8,gs=quantize_rows(weighted)
        expected=(g8.float()*gs)@q.weight_fp8.float()
        torch.testing.assert_close(x.grad,expected)
        dense=g@(q.weight_fp8.float()*q.weight_scale)
        self.assertLess(float((x.grad-dense).norm()/dense.norm()),.06)
    def test_three_updates_reach_early_and_late_LoRA(self):
        one=FP8StyleLinear(StyleLinear(self.base(),rank=4,alpha=4),reference=True)
        two=FP8StyleLinear(StyleLinear(self.base(),rank=4,alpha=4),reference=True)
        model=nn.Sequential(one,nn.SiLU(),two);p=[one.style_A,one.style_B,two.style_A,two.style_B]
        opt=torch.optim.AdamW(p,lr=1e-3)
        for step in range(3):
            opt.zero_grad();model(torch.randn(19,64)).square().mean().backward()
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in p))
            self.assertTrue(all(m.style_B.grad.norm()>0 for m in [one,two]))
            if step:self.assertTrue(all(m.style_A.grad.norm()>0 for m in [one,two]))
            opt.step()
        self.assertTrue(all(t.dtype==torch.float32 for t in p))
        self.assertTrue(all(not t.requires_grad for m in [one,two] for t in m.base.buffers()))
    def test_zero_style_matches_quantized_base_exactly(self):
        q=FP8StyleLinear(StyleLinear(self.base(),rank=4,alpha=4),reference=True);x=torch.randn(19,64)
        self.assertTrue(torch.equal(q(x),q.base(x)))
    def test_quantization_never_modifies_source_weights(self):
        b=self.base();original=b.weight.clone();FP8FrozenLinear(b,reference=True)
        self.assertTrue(torch.equal(b.weight,original))
