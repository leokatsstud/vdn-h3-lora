"""Frozen W8A8 linears with an explicit input-gradient backward for LoRA.

Only the frozen base is quantized. LoRA masters/Adam stay FP32; norms, modulation,
gates, heads, and narrow projections stay at their original precision. The FP8
backward is a straight-through approximation, not a derivative of rounding.
CPU reference mode exists only for tiny tests and must never be selected by a GPU job.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .adapter import StyleLinear

DTYPE=torch.float8_e4m3fn
MAX=torch.finfo(DTYPE).max

def quantize_rows(x):
    if x.is_cuda:
        from src.models.ops.fp8_linear import quantize_rows as pinned_quantizer
        return pinned_quantizer(x.detach().contiguous())
    scale=(x.detach().float().abs().amax(dim=1,keepdim=True)/MAX).clamp_min(1e-12)
    return (x.detach().float()/scale).clamp(-MAX,MAX).to(DTYPE).contiguous(),scale.contiguous()

def gemm(a,b,sa,sb,dtype,reference):
    if reference:
        return ((a.float()*sa)@(b.float()*sb)).to(dtype)
    if not a.is_cuda or torch.cuda.get_device_capability(a.device)[0]!=9:
        raise RuntimeError('This FP8 training path is explicitly validated for Hopper only; no silent fallback')
    if a.shape[1]%16 or b.shape[1]%16:raise ValueError('FP8 GEMM dimensions must be aligned to16')
    return torch._scaled_mm(a,b,scale_a=sa,scale_b=sb,out_dtype=dtype,use_fast_accum=False)

class W8A8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w8,ws,bias,reference):
        if x.ndim!=2:raise ValueError('Expected flat rows')
        ctx.save_for_backward(w8,ws)
        ctx.reference=bool(reference);ctx.dtype=x.dtype;ctx.rows=x.shape[0]
        pad=(-x.shape[0])%16
        xp=F.pad(x,(0,0,0,pad)) if pad else x
        x8,xs=quantize_rows(xp)
        y=gemm(x8,w8.T,xs,ws.T,x.dtype,ctx.reference)[:ctx.rows]
        return y if bias is None else y+bias
    @staticmethod
    def backward(ctx,dy):
        if any(ctx.needs_input_grad[1:]):raise RuntimeError('Only input gradients are supported; base weight/bias must be frozen')
        if not ctx.needs_input_grad[0]:return None,None,None,None,None
        w8,ws=ctx.saved_tensors
        # W = diag(ws) W8. Its row scale is on the contracted dimension of dX,
        # so absorb it into dY BEFORE quantization; scaling dX afterward is wrong.
        z=dy.detach().float()*ws.T
        pad=(-z.shape[0])%16
        if pad:z=F.pad(z,(0,0,0,pad))
        z8,zs=quantize_rows(z)
        # scaled_mm needs a column-major B. Materialize only this layer's FP8
        # transpose during backward, never a permanent second copy of the model.
        wt=w8.T.contiguous().T
        unit=torch.ones(1,w8.shape[1],device=dy.device,dtype=torch.float32)
        dx=gemm(z8,wt,zs,unit,ctx.dtype,ctx.reference)[:ctx.rows]
        return dx,None,None,None,None

class FP8FrozenLinear(nn.Module):
    def __init__(self,base,reference=False):
        super().__init__()
        if base.weight.requires_grad or (base.bias is not None and base.bias.requires_grad):raise ValueError('Base must be frozen before quantization')
        self.in_features=base.in_features;self.out_features=base.out_features
        self.reference=bool(reference)
        w,s=quantize_rows(base.weight)
        self.register_buffer('weight_fp8',w);self.register_buffer('weight_scale',s)
        self.register_buffer('bias',None if base.bias is None else base.bias.detach())
    def forward(self,x):
        y=W8A8Function.apply(x.reshape(-1,x.shape[-1]),self.weight_fp8,self.weight_scale,self.bias,self.reference)
        return y.reshape(*x.shape[:-1],self.out_features)

class FP8StyleLinear(StyleLinear):
    def __init__(self,old,reference=False):
        nn.Module.__init__(self)
        self.base=FP8FrozenLinear(old,reference)
        for name in ['in_features','out_features','rank','alpha','strength']:setattr(self,name,getattr(old,name))
        self.style_A=old.style_A;self.style_B=old.style_B
    def forward(self,x):
        y=self.base(x)
        if self.strength==0:return y
        delta=F.linear(F.linear(x,self.style_A.to(x.dtype)),self.style_B.to(x.dtype))
        return y+delta*(self.strength*self.alpha/self.rank)

def convert_frozen_linears(model,min_width=4096,reference=False):
    replacements=[];old_bytes=0;new_bytes=0
    # Keep names, not references to every old BF16 module: otherwise conversion
    # retains the whole BF16 base alongside the new FP8 base until the loop ends.
    candidates=[path for path,module in model.named_modules() if isinstance(module,(nn.Linear,StyleLinear))]
    for path in candidates:
        module=model.get_submodule(path)
        if not path.startswith('transformer_blocks.'):continue
        # Adaptive gates/AdaLN/refiners/norms/heads intentionally remain original.
        eligible=any(p in path for p in ['.attn.orig.to_q','.attn.orig.to_k','.attn.orig.to_v','.attn.orig.to_out.0','.attn.to_q','.attn.to_k','.attn.to_v','.attn.to_out.0','.ff.net.0.proj','.ff.net.2','.attn.to_out_linear'])
        if not eligible or not isinstance(module,(nn.Linear,StyleLinear)):continue
        if min(module.in_features,module.out_features)<min_width:continue
        if module.in_features%16 or module.out_features%16:raise ValueError('Unaligned target: '+path)
        replacement=FP8StyleLinear(module,reference) if isinstance(module,StyleLinear) else FP8FrozenLinear(module,reference)
        base=replacement.base if isinstance(replacement,FP8StyleLinear) else replacement
        old_bytes+=module.weight.numel()*module.weight.element_size()
        new_bytes+=base.weight_fp8.numel()*base.weight_fp8.element_size()+base.weight_scale.numel()*4
        parent,name=path.rsplit('.',1);setattr(model.get_submodule(parent),name,replacement)
        replacements.append(path)
    if not replacements:raise ValueError('No eligible frozen linears were quantized')
    return {'scheme':'frozen W8A8 e4m3 rowwise forward + scaled-input-gradient backward','linears':len(replacements),'paths':replacements,'old_selected_weight_bytes':old_bytes,'new_selected_weight_bytes':new_bytes,'backward':'explicit FP8 dX; no base dW','LoRA_masters':'FP32','cpu_reference':reference,'inference_only_kernels':False}
