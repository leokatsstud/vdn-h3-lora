import contextlib,math,re
import torch
from torch import nn
from torch.nn import functional as F

TARGET=re.compile(r'^transformer_blocks\.\d+\.(attn\.(?:orig\.)?(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))$')

class StyleLinear(nn.Module):
    """Keep frozen effective W at its original key; FP32 style master tensors only."""
    def __init__(self,base,rank=64,alpha=64):
        super().__init__()
        if not isinstance(base,nn.Linear):raise TypeError('Style target must be an unwrapped nn.Linear')
        self.weight=base.weight;self.bias=base.bias
        self.in_features=base.in_features;self.out_features=base.out_features
        self.rank=rank;self.alpha=alpha;self.strength=1.0
        self.weight.requires_grad_(False)
        if self.bias is not None:self.bias.requires_grad_(False)
        self.style_A=nn.Parameter(torch.empty(rank,self.in_features,device=base.weight.device,dtype=torch.float32))
        self.style_B=nn.Parameter(torch.zeros(self.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.style_A,a=math.sqrt(5))
    def forward(self,x):
        y=F.linear(x,self.weight,self.bias)
        if self.strength==0:return y
        delta=F.linear(F.linear(x,self.style_A.to(x.dtype)),self.style_B.to(x.dtype))
        return y+delta*(self.strength*self.alpha/self.rank)

def install_style(model,rank=64,alpha=64,expected_modules=200,expected_parameters=160563200,target_suffixes=None):
    model.requires_grad_(False);all_paths=[n for n,m in model.named_modules() if TARGET.fullmatch(n)]
    paths=[n for n in all_paths if target_suffixes is None or any(n.endswith('.'+suffix) for suffix in target_suffixes)]
    if len(paths)!=expected_modules:raise ValueError(f'Style targets: {len(paths)} != {expected_modules}')
    for path in all_paths:
        parent,name=path.rsplit('.',1);owner=model.get_submodule(parent);base=owner.get_submodule(name)
        if path not in paths:
            # Consume the exact original A-initialization draws even for omitted
            # targets. Common FF/V/O masters then start identically across scopes.
            scratch=torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32)
            nn.init.kaiming_uniform_(scratch,a=math.sqrt(5));del scratch
            continue
        setattr(owner,name,StyleLinear(base,rank,alpha))
    count=sum(p.numel() for p in style_parameters(model))
    if expected_parameters is not None and count!=expected_parameters:raise ValueError(f'Style count {count} != {expected_parameters}')
    for name,p in model.named_parameters():
        if p.requires_grad != (name.endswith('.style_A') or name.endswith('.style_B')):raise ValueError('Unexpected trainable: '+name)
    return paths
def style_modules(model):return [(n,m) for n,m in model.named_modules() if isinstance(m,StyleLinear)]
def style_parameters(model):return [p for _,m in style_modules(model) for p in (m.style_A,m.style_B)]
@contextlib.contextmanager
def style_strength(model,strength):
    modules=style_modules(model);old=[m.strength for _,m in modules]
    try:
        for _,m in modules:m.strength=float(strength)
        yield
    finally:
        for (_,m),value in zip(modules,old):m.strength=value
def style_state(model):return {n: p.detach().float().cpu().clone() for n,p in model.named_parameters() if n.endswith(('.style_A','.style_B'))}
def load_style_state(model,state):
    targets={n:p for n,p in model.named_parameters() if n.endswith(('.style_A','.style_B'))}
    if set(state)!=set(targets):raise ValueError('Style checkpoint keys differ from current target set')
    with torch.no_grad():
        for n,p in targets.items():
            if p.shape!=state[n].shape:raise ValueError('Style shape mismatch: '+n)
            p.copy_(state[n].to(p.device,p.dtype))
def export_style(model,path,metadata):
    from safetensors.torch import save_file
    import os
    tensors={}
    for n,m in style_modules(model):
        tensors[n+'.lora_A.weight']=m.style_A.detach().to('cpu',torch.bfloat16).contiguous()
        tensors[n+'.lora_B.weight']=m.style_B.detach().to('cpu',torch.bfloat16).contiguous()
    tmp=str(path)+'.tmp';save_file(tensors,tmp,metadata={k:str(v) for k,v in metadata.items()});os.replace(tmp,path)
@torch.no_grad()
def merge_style(model,strength):
    for n,m in style_modules(model):
        m.weight.add_(((m.style_B.float()@m.style_A.float())*(strength*m.alpha/m.rank)).to(m.weight.dtype))
        linear=nn.Linear(m.in_features,m.out_features,bias=m.bias is not None,device='meta',dtype=m.weight.dtype)
        linear.weight=m.weight;linear.bias=m.bias
        parent,name=n.rsplit('.',1);setattr(model.get_submodule(parent),name,linear)
