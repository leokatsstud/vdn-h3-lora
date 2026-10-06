"""Read-only checkpoint benchmark: identical data/noise, no optimizer updates."""
import argparse,contextlib,gc,inspect,json,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from stylelib.common import contract,read_json,write_json,fingerprint
from stylelib.model import build_style_model
from stylelib.adapter import load_style_state,style_parameters,style_strength,style_modules
from stylelib.packing import PackedCondition,noise_sample,predict
from stylelib.objective import paired_times,guidance_target,style_loss

ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['current','no_offload','block_only','block_only_no_offload'],required=True);ap.add_argument('--fit-only',action='store_true');ap.add_argument('--cases',nargs='+',default=['V145','V114']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args()
base=a.root;out=base/'work/perf_r32_step100';out.mkdir(exist_ok=True)
cfg=contract();index=read_json(base/'work/cache_train_175/index.json')
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
assert saved['step']==100 and saved['config']==cfg
identity=saved['identity'];masters=saved['style'];del saved
torch.manual_seed(20261002)
model,info=build_style_model(base/'work/weights',cfg,'cuda:0',True,'flex');load_style_state(model,masters);del masters;gc.collect()
if a.variant.startswith('block_only'):
    for block in model.transformer_blocks:
        for name in ('attn','ff'):
            module=getattr(block,name)
            inner=inspect.getclosurevars(module.forward).nonlocals.get('inner')
            if not callable(inner):raise ValueError('Unexpected checkpoint wrapper; cannot benchmark safely')
            module.forward=inner;module._checkpointed=False
offload=a.variant in ('current','block_only')
optimizer_reservation=[torch.empty_like(p).zero_() for p in style_parameters(model) for _ in range(2)]

def measure(entry):
    record=entry['record'];media=torch.load(base/'work/cache_train_175'/entry['media'],map_location='cpu',weights_only=True)
    text=torch.load(base/'work/cache_train_175'/entry['conditions']['t2v'],map_location='cpu',weights_only=True)
    assert media['key']==text['key']==entry['key']
    shape=tuple(media['video_latents'].shape);af=media['audio_latents'].shape[-1]
    pack=PackedCondition(dict(text['conditional'],condition_latents=[]),shape,af,'cuda:0')
    unpack=PackedCondition(dict(text['unconditional'],condition_latents=[]),shape,af,'cuda:0')
    rng=torch.Generator(device='cuda:0').manual_seed(424242)
    tv,ta=paired_times(.4,cfg['loss']['video_shift'],cfg['loss']['audio_shift'])
    vr,ar,data_v,data_a,refs=noise_sample(media,pack.refs,tv,ta,rng,'cuda:0')
    for p in style_parameters(model):p.grad=None
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.no_grad(),style_strength(model,0):base_v,base_a=predict(model,pack,vr,ar,tv,ta,refs)
    with torch.no_grad(),style_strength(model,1):un_v,un_a=predict(model,unpack,vr,ar,tv,ta,refs)
    target_v=guidance_target(data_v,un_v,cfg['loss']['guidance_scale_video']);target_a=guidance_target(data_a,un_a,cfg['loss']['guidance_scale_audio'])
    mask_v=media['video_valid_frames'].repeat_interleave((shape[2]//2)*(shape[3]//2));mask_a=media['audio_valid_rows']
    context=torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
    with style_strength(model,1),context:
        pred_v,pred_a=predict(model,pack,vr,ar,tv,ta,refs)
        loss,_=style_loss(pred_v,pred_a,target_v,target_a,base_v,base_a,mask_v,mask_a,record['audio_supervised'],cfg['loss'])
        torch.cuda.synchronize();forward=time.perf_counter()-start
        loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-start
    result={'status':'ok','variant':a.variant,'id':record['id'],'frames':record['frames'],'seconds':seconds,'forward_seconds':forward,'backward_seconds':seconds-forward,'loss':float(loss.detach()),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'checkpoint_identity':identity,'checkpoint_step':100,'config_sha256':fingerprint(cfg),'noise_seed':424242,'base_u':.4,'optimizer_updates':0,'offload':offload,'checkpoint_parts':'block_only' if a.variant.startswith('block_only') else 'block_attention_ff'}
    params={n:p for n,p in model.named_parameters() if n.endswith(('.style_A','.style_B'))}
    assert len(params)==600 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    reference=out/(record['id']+'_reference_gradients.safetensors')
    if a.fit_only:result['fit_only']=True;result['optimizer_memory_reserved']=True
    elif a.variant=='current':save_file({n:p.grad.detach().cpu().contiguous() for n,p in params.items()},str(reference))
    else:
        old=load_file(str(reference),device='cpu');difference=norm=dot=newnorm=0.
        for n,p in params.items():
            g=p.grad.detach().cpu().double();r=old[n].double();difference+=float((g-r).square().sum());norm+=float(r.square().sum());dot+=float((g*r).sum());newnorm+=float(g.square().sum())
        result['gradient_relative_l2']=(difference/max(norm,1e-30))**.5
        result['gradient_cosine']=dot/max((norm*newnorm)**.5,1e-30)
        baseline=read_json(out/(record['id']+'_current.json'))
        result['relative_loss_difference']=abs(result['loss']-baseline['loss'])/max(abs(baseline['loss']),1e-12)
        result['parity_passed']=result['gradient_relative_l2']<.001 and result['gradient_cosine']>.99999 and result['relative_loss_difference']<.0001
    return result

for case in a.cases:
    entry=next(e for e in index['records'] if e['record']['id']==case)
    try:result=measure(entry)
    except torch.cuda.OutOfMemoryError:
        result={'status':'oom','variant':a.variant,'id':case,'checkpoint_step':100,'optimizer_updates':0}
    write_json(out/(case+'_'+a.variant+'.json'),result);print(json.dumps(result),flush=True)
    for p in style_parameters(model):p.grad=None
    gc.collect();torch.cuda.empty_cache()
