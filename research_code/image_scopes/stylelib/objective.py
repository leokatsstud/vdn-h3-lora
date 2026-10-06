import torch

def paired_times(u,video_shift=12.,audio_shift=3.):
    u=float(u)
    if not 0<u<=1:raise ValueError('u outside (0,1]')
    return 1-video_shift*u/(1+(video_shift-1)*u),1-audio_shift*u/(1+(audio_shift-1)*u)
def guidance_target(data,uncond,scale):return data if uncond is None else uncond+scale*(data-uncond)
def masked_mse(pred,target,mask):
    mask=mask.to(device=pred.device,dtype=torch.bool)
    if mask.ndim!=1 or len(mask)!=len(pred):raise ValueError('Row mask shape differs from prediction')
    if not bool(mask.any()):return pred.sum()*0
    return (pred.float()[mask]-target.detach().float()[mask]).square().mean()
def style_loss(v,a,target_v,target_a,base_v,base_a,mask_v,mask_a,audio_supervised,cfg):
    terms={'video_data':masked_mse(v,target_v,mask_v),
           'video_preservation':masked_mse(v,base_v,mask_v),
           'audio_preservation':masked_mse(a,base_a,mask_a)}
    terms['audio_data']=masked_mse(a,target_a,mask_a) if audio_supervised else a.sum()*0
    total=(cfg['video_data_weight']*terms['video_data']+cfg['video_preservation_weight']*terms['video_preservation']+
           cfg['clean_audio_data_weight']*terms['audio_data']+cfg['audio_preservation_weight']*terms['audio_preservation'])
    return total,terms
