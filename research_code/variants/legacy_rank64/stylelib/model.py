import torch
from .adapter import install_style
from .common import read_json

def normalize_hybrid_storage(model,predicate):
    """Replace FSDP's mixed-precision parameter casting for this unsharded trainer."""
    counts={'bf16':0,'fp32_alpha':0}
    for name,p in model.named_parameters():
        if predicate(name):
            alpha='.linear_attention.alpha.' in name
            p.data=p.data.to(torch.float32 if alpha else torch.bfloat16)
            counts['fp32_alpha' if alpha else 'bf16']+=1
    return counts

def build_style_model(weights,config,device='cuda:0',checkpointing=True,backend='flex'):
    from src.checkpoints import load_checkpoint
    from src.models.factory import build_model,load_model_weights
    from src.models.hybrid_transform import set_inference_mode,set_softmax_backend,is_transform_parameter,iter_hybrids
    from src.inference.utils.lora import merge_lora_state
    from src.utils.activation_checkpointing import split_block_checkpoints
    from pathlib import Path
    weights=Path(weights).resolve();lock=read_json(weights/'sources.json')
    verified=read_json(weights/'verified.json')
    if not verified.get('vdn_files_verified') or verified.get('sources')!=lock:raise ValueError('Run download_weights.py --execute --verify before model loading')
    if lock['vdn_revision']!=config['hf_revision']:raise ValueError('Wrong VDN weight revision')
    path=weights/'vdn'/config['checkpoint'];art=load_checkpoint(str(path))
    if art.metadata.get('truncated_blocks'):raise ValueError('Truncated checkpoint forbidden')
    adapters={a['config'].get('name','default'):a['config'] for a in art.model_spec['adapters']}
    if set(adapters)!=set(config['active_frozen_adapters']):raise ValueError('VDN adapters differ from default + turbo')
    for a in adapters.values():
        if a['rank']!=a['alpha'] or dict(a.get('rank_pattern',{}))!=dict(a.get('alpha_pattern',{})):
            raise ValueError('Pinned merge recipe requires alpha/r=1 for every shipped adapter')
    model=build_model(art.model_spec,device='cpu',base_source=str(weights/'vdn/h3-base'))
    # Fresh hybrid linears start FP32. FSDP normally casts them, but this trainer
    # is unsharded: match BF16 hidden states while retaining the alpha FP32 island.
    storage=normalize_hybrid_storage(model,is_transform_parameter)
    branch={k:v for k,v in art.weights.items() if 'lora_' not in k}
    expected={n for n,_ in model.named_parameters() if is_transform_parameter(n)}
    if expected!=set(branch):raise ValueError(f'VDN branch incomplete: missing {sorted(expected-set(branch))[:8]} extra {sorted(set(branch)-expected)[:8]}')
    loaded=load_model_weights(model,branch)
    loras={k:v for k,v in art.weights.items() if 'lora_' in k}
    pairs=merge_lora_state(model,loras)
    if pairs*2!=len(loras):raise ValueError('Unpaired or unloaded shipped LoRA tensors')
    del branch,loras,art
    # Preserve upstream FP32 alpha masters; do NOT call model.to(dtype=bf16).
    model.to(device);set_inference_mode(model,False);set_softmax_backend(model,backend)
    for attn in iter_hybrids(model):attn.teacher_mode=False
    paths=install_style(model,config['style']['rank'],config['style']['alpha'],config['style']['expected_modules'],config['style']['expected_parameters'])
    if checkpointing:
        split_block_checkpoints(model);model.enable_gradient_checkpointing()
    # Diffusers checkpoints whenever grads are enabled, independent of .training.
    # Eval disables any stochastic dropout for baseline-preservation comparisons.
    model.eval()
    return model,{'branch_tensors':loaded,'branch_storage':storage,'frozen_merged_adapter_pairs':pairs,'style_modules':len(paths),'vdn_revision':lock['vdn_revision'],'text_revision':lock['text_revision'],'softmax_backend':backend,'inference_kernels':False}
