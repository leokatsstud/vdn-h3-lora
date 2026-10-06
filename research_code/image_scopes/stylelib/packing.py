"""Dynamic geometry, shared noising, ref rows excluded from target and linear scan."""
from types import SimpleNamespace
import torch

PATCH=(1,2,2)
def patchify(z):
    b,c,t,h,w=z.shape
    if b!=1 or h%2 or w%2:raise ValueError('Expected batch 1, even latent H/W')
    return z.reshape(b,c,t,h//2,2,w//2,2).permute(0,2,3,5,1,4,6).reshape(-1,c*4)
def unpatchify(rows,shape):
    c,t,h,w=shape
    return rows.reshape(1,t,h//2,w//2,c,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,c,t,h,w)

class PackedCondition:
    def __init__(self,condition,video_shape,audio_frames,device):
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep,MiniMaxH3Ref2VAPrepareLayoutStep
        from src.models.sequence_layout import layout_from_indices
        self.device=device;self.video_shape=tuple(video_shape);c,t,h,w=video_shape
        self.embeds=condition['prompt_embeds'].to(device,torch.bfloat16)[None]
        tags=condition['text_token_tags'];refs=condition.get('condition_latents',[])
        self.refs=[z.to(device,torch.float32) for z in refs]
        if refs:
            modalities=[SimpleNamespace(kind='image',has_audio=False) for _ in refs]
            layout=MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(tags,modalities,refs,[],t,h,w,audio_frames,PATCH,2,2,0)
        else:
            layout=MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags,t,h,w,audio_frames,PATCH,2,2,0,keyframe_anchors=())
        pos,tags,vi,ai,ti,ncv,nca=layout
        if nca:raise ValueError('Audio references unsupported in this image-reference pilot')
        self.pos,self.tags,self.vi,self.ai,self.ti=[x.to(device) for x in (pos,tags,vi,ai,ti)]
        self.ncv=int(ncv);self.seq_len=len(pos)
        self.layout=layout_from_indices(self.vi[ncv:],t,(h//2)*(w//2),seq_len=self.seq_len,frame_size=(h//2,w//2),text_indices=self.ti)
        if len(self.vi)-ncv!=t*(h//2)*(w//2):raise ValueError('Target layout geometry mismatch')
    def inputs(self,video_rows,audio_rows,tv,ta,ref_rows):
        if sum(len(z) for z in ref_rows)!=self.ncv:raise ValueError('Reference rows mismatch')
        vr=torch.cat([*ref_rows,video_rows]) if ref_rows else video_rows
        times=torch.full((self.seq_len,),float(tv),dtype=torch.float32,device=self.device)
        times[self.ai]=float(ta)
        # Pinned conditions at 0.999 in both conditional and near-empty layouts.
        if self.ncv:times[self.vi[:self.ncv]]=0.999
        ts,indices=torch.unique(times,sorted=True,return_inverse=True)
        return dict(hidden_states=vr[None],audio_hidden_states=audio_rows[None],encoder_hidden_states=self.embeds,
                    timestep=ts,timestep_indices=indices,token_tags=self.tags,position_ids=self.pos,
                    video_indices=self.vi,audio_indices=self.ai,text_indices=self.ti,return_dict=False)

def noise_sample(media,refs,tv,ta,generator,device):
    # Same order as renderer: reference noise before generated video/audio noise.
    ref_rows=[]
    for z in refs:
        noise=torch.randn(z.shape,device=device,generator=generator,dtype=torch.float32)
        ref_rows.append(patchify(0.999*z+0.001*noise))
    v=media['video_latents'].to(device,torch.float32)[None];a=media['audio_latents'].to(device,torch.float32)
    a=a.permute(0,2,1).reshape(-1,a.shape[1])
    ev=torch.randn(v.shape,device=device,generator=generator,dtype=torch.float32)
    ea=torch.randn(a.shape,device=device,generator=generator,dtype=torch.float32)
    return patchify(tv*v+(1-tv)*ev),ta*a+(1-ta)*ea,patchify(v-ev),a-ea,ref_rows

def predict(model,packed,vr,ar,tv,ta,refs):
    from src.models.hybrid_transform import set_layout
    set_layout(model,packed.layout)
    v,a=model(**packed.inputs(vr,ar,tv,ta,refs))
    return v[0,packed.ncv:].float(),a[0].float()
