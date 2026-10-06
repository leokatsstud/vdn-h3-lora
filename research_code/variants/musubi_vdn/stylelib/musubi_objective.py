"""Musubi guidance target + video-only MSE. No base-preservation losses."""
import torch
def paired_times(u,video_shift=12.,audio_shift=3.):
    u=float(u)
    if not 0<u<=1:raise ValueError('u outside (0,1]')
    return 1-video_shift*u/(1+(video_shift-1)*u),1-audio_shift*u/(1+(audio_shift-1)*u)
def guidance_target(data,uncond,scale):return data if uncond is None else uncond+scale*(data-uncond)
def masked_mse(pred,target,mask):
    mask=mask.to(device=pred.device,dtype=torch.bool)
    if mask.ndim!=1 or len(mask)!=len(pred):raise ValueError('Prediction mask shape differs')
    if not bool(mask.any()):raise ValueError('Image target has no supervised video rows')
    return (pred.float()[mask]-target.detach().float()[mask]).square().mean()
def style_loss(v,a,target_v,target_a,base_v,base_a,mask_v,mask_a,audio_supervised,cfg):
    if audio_supervised:raise ValueError('Image recipe forbids target audio supervision')
    if any(cfg[k]!=0 for k in ['video_preservation_weight','audio_preservation_weight','clean_audio_data_weight']):raise ValueError('Unexpected additional loss')
    loss=masked_mse(v,target_v,mask_v)
    return loss,{'video_data':loss,'video_preservation':loss.detach()*0,'audio_preservation':loss.detach()*0,'audio_data':loss.detach()*0}
