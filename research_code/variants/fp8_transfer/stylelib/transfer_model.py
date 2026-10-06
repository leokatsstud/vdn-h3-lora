"""Explicit original-H3 / VDN switch, never teacher_mode on a merged VDN base."""
import copy,torch
from pathlib import Path
from .common import read_json
from .adapter import install_style
from .model import build_style_model as build_vdn
from .fp8_training import convert_frozen_linears

def build_style_model(weights,config,device='cuda:0',checkpointing=True,backend='flex',architecture='vdn',precision='bf16'):
    if architecture=='vdn':
        model,info=build_vdn(weights,config,device,checkpointing,backend)
    elif architecture=='dense':
        from src.checkpoints import load_checkpoint
        from src.models.factory import build_model
        from src.utils.activation_checkpointing import split_block_checkpoints
        weights=Path(weights);lock=read_json(weights/'sources.json');verified=read_json(weights/'verified.json')
        if verified.get('sources')!=lock or not verified.get('vdn_files_verified'):raise ValueError('Unverified original-H3 source')
        artifact=load_checkpoint(str(weights/'vdn'/config['checkpoint']))
        spec=copy.deepcopy(artifact.model_spec);spec['transforms']=[];spec['adapters']=[]
        del artifact
        model=build_model(spec,device='cpu',base_source=str(weights/'vdn/h3-base'))
        model.to(device)
        paths=install_style(model,config['style']['rank'],config['style']['alpha'],config['style']['expected_modules'],config['style']['expected_parameters'],target_suffixes=config['style']['target_suffixes'])
        if checkpointing:split_block_checkpoints(model);model.enable_gradient_checkpointing()
        model.eval()
        info={'style_modules':len(paths),'frozen_merged_adapter_pairs':0,'active_frozen_adapters':[],'vdn_revision':lock['vdn_revision'],'text_revision':lock['text_revision'],'original_base_spec':spec['base'],'inference_kernels':False,'softmax_backend':'original dense attention'}
    else:raise ValueError('Unknown architecture')
    info['architecture']=architecture;info['training_precision']=precision
    info['resident_parameter_and_buffer_bytes_before']=sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
    if precision=='fp8':
        info['FP8']=convert_frozen_linears(model)
        torch.cuda.empty_cache()
    elif precision!='bf16':raise ValueError('Unknown precision')
    info['resident_parameter_and_buffer_bytes_after']=sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
    return model,info
