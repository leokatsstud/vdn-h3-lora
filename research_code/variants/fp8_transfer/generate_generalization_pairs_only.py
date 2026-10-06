"""Twelve generation-only videos on one loaded VDN. No training, transfer or rental actions."""
import argparse,gc,time,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from stylelib.common import read_json,write_json,contract,sha256
from stylelib.image_recipe import apply_image_recipe
from stylelib.transfer_model import build_style_model
from stylelib.adapter import style_modules
from stylelib.packing import PackedCondition,patchify,unpatchify,predict

class AdditiveInferenceStyle(nn.Module):
 """Official frozen inference Linear + separate FP32 masters cast to BF16 for LoRA."""
 def __init__(self,base,rank=32,weight_shape=None):
  super().__init__();self.base=base
  if weight_shape is None:self.in_features=base.in_features;self.out_features=base.out_features
  else:self.out_features,self.in_features=weight_shape
  self.strength=1.0
  reference=next(base.parameters(),None)
  if reference is None:reference=next(base.buffers())
  self.style_A=nn.Parameter(torch.zeros(rank,self.in_features,device=reference.device,dtype=torch.float32),requires_grad=False)
  self.style_B=nn.Parameter(torch.zeros(self.out_features,rank,device=reference.device,dtype=torch.float32),requires_grad=False)
 def forward(self,x):
  return self.base(x)+F.linear(F.linear(x,self.style_A.to(x.dtype)),self.style_B.to(x.dtype))*self.strength

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--merge-controls',action='store_true');a=ap.parse_args();root=a.root;work=root/'work';weights=work/'weights';out=work/'generalization_20261006';out.mkdir(exist_ok=True)
 expected_total=14 if a.merge_controls else 12
 suite=read_json(root/'private/generalization_suite.json');cache=work/'generalization_cache_20261006';index=read_json(cache/'index.json');assert index['complete'] and index['suite']==suite
 assert suite['required_videos']==12 and len(suite['branches'])==2 and len(suite['cases'])==6
 adapters={b['id']:Path(b['server_adapter_path']) for b in suite['branches']}
 for b in suite['branches']:assert sha256(adapters[b['id']])==b['adapter_sha256']
 from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio,MiniMaxH3Scheduler
 from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames,audio_latent_num_frames
 from src.inference.render import decode_and_save
 from src.models.hybrid_transform import set_inference_mode
 torch.set_grad_enabled(False);results=[]
 def progress(stage,**kw):write_json(out/'progress.json',{'stage':stage,'completed':len(results),'total':expected_total,'time_epoch':time.time(),**kw})
 progress('load_VDN_once')
 from omegaconf import OmegaConf
 from stylelib import GPU_ROOT
 from stylelib.adapter import TARGET
 from src.config.inference import InferenceConfig
 from src.inference.utils.assemble import build_inference_model,render_record
 cfg=OmegaConf.merge(OmegaConf.structured(InferenceConfig),OmegaConf.load(str(GPU_ROOT/'upstream/configs/inference/8nfe_tuned_fp8.yaml')),{'checkpoint':str(weights/'vdn/stage-dmd-step-250'),'base_source':str(weights/'vdn/h3-base'),'vae_source':str(weights/'vdn/h3-base'),'external_loras':[],'precision':{'fp8':{'enabled':not a.merge_controls}},'render':{'prompt_file':'cached','out':'pending','num_steps':8,'num_frames':345,'warmup_steps':0,'device':'cuda:0'}})
 bundle=build_inference_model(cfg,'cuda:0');assert bundle.is_hybrid
 pristine={}
 if a.merge_controls:
  progress('save_pristine_base_CPU_for_merge_controls')
  for name,layer in bundle.transformer.named_modules():
   if isinstance(layer,nn.Linear) and min(layer.in_features,layer.out_features)>=4096:
    pristine[name]=(layer.weight.detach().cpu(),None if layer.bias is None else layer.bias.detach().cpu())
  assert len(pristine)==363
  from src.models.ops.fp8_linear import convert_linear_to_fp8
  bundle.fp8_linears=convert_linear_to_fp8(bundle.transformer,skip_end_blocks=0);cfg.precision.fp8.enabled=True
 assert bundle.fp8_linears==363
 model=bundle.transformer;info=render_record(cfg,bundle);info['FP8']={'linears':bundle.fp8_linears,'scheme':'official 8nfe_tuned_fp8 frozen base; separate BF16 style residual'}
 candidates=[(n,m) for n,m in model.named_modules() if TARGET.match(n)];assert len(candidates)==300
 modules=[]
 for name,base in candidates:
  shape=pristine[name][0].shape if name in pristine else None
  module=AdditiveInferenceStyle(base,weight_shape=shape);owner,leaf=name.rsplit('.',1);setattr(model.get_submodule(owner),leaf,module);modules.append((name,module))
 del candidates
 model.requires_grad_(False);vae=bundle.vae;audio_vae=bundle.audio_vae;vae.enable_tiling()
 counts={n:0 for n,_ in modules};hooks=[]
 for name,m in modules:
  def hook(module,inputs,output,key=name):counts[key]+=1
  hooks.append(m.register_forward_hook(hook))
 for case in suite['cases']:
  assert not case['references'] and case['frames']==345 and case['fps']==24
  pair=[]
  for branch in suite['branches']:
   if (root/'GLOBAL_STOP').exists():raise InterruptedError('Stop requested before next video')
   bid=branch['id'];dest=out/case['id']/bid;dest.mkdir(parents=True,exist_ok=True);video=dest/(case['id']+f'_seed{case["seed"]}.mp4')
   if video.exists() and video.with_suffix('.json').exists():
    meta=read_json(video.with_suffix('.json'));assert meta['video_sha256']==sha256(video) and meta['adapter_sha256']==branch['adapter_sha256'] and meta['prompt']==case['prompt'] and meta['seed']==case['seed'];results.append(meta);pair.append(meta);continue
   progress('load_style',case=case['id'],branch=bid)
   with safe_open(str(adapters[bid]),framework='pt',device='cpu') as f:
    keys=set(f.keys());assert len(keys)==600
    for name,m in modules:
     source=name if name+'.lora_A.weight' in keys else name.replace('.attn.orig.','.attn.')
     A=f.get_tensor(source+'.lora_A.weight');B=f.get_tensor(source+'.lora_B.weight');assert A.shape==m.style_A.shape and B.shape==m.style_B.shape
     m.style_A.copy_(A.to('cuda:0',torch.float32));m.style_B.copy_(B.to('cuda:0',torch.float32));m.strength=1
   payload=torch.load(cache/(case['id']+'.pt'),map_location='cpu',weights_only=True);assert payload['key']==index['key'] and payload['case']==case
   t=video_latent_num_frames(case['frames'],17,5);af=audio_latent_num_frames(case['frames']);shape=(model.config.in_channels,t,case['height']//16,case['width']//16)
   pack=PackedCondition({'prompt_embeds':payload['prompt_embeds'],'text_token_tags':payload['text_token_tags'],'condition_latents':payload['condition_latents']},shape,af,'cuda:0');assert not pack.refs
   gen=torch.Generator('cuda:0').manual_seed(case['seed']);vn=torch.randn((1,*shape),generator=gen,device='cuda:0');ar=torch.randn((af*2,32),generator=gen,device='cuda:0')
   noise={'video':hashlib.sha256(vn.cpu().numpy().tobytes()).hexdigest(),'audio':hashlib.sha256(ar.cpu().numpy().tobytes()).hexdigest()};vr=patchify(vn);del vn
   vs=MiniMaxH3Scheduler(shift=12);aus=MiniMaxH3Scheduler(shift=3);vs.set_timesteps(9,device='cuda:0');aus.set_timesteps(9,device='cuda:0');times=[]
   for name in counts:counts[name]=0
   progress('generate',case=case['id'],branch=bid,denoise_step=0)
   started=time.perf_counter()
   for step,(tv,ta) in enumerate(zip(vs.timesteps,aus.timesteps)):
    if (root/'GLOBAL_STOP').exists():raise InterruptedError('Stop requested at denoise boundary')
    torch.cuda.synchronize();s=time.perf_counter();v,audio=predict(model,pack,vr,ar,float(tv),float(ta),[]);vr=vs.step(v,tv,vr,return_dict=False)[0];ar=aus.step(audio,ta,ar,return_dict=False)[0];torch.cuda.synchronize();times.append(time.perf_counter()-s)
    progress('generate',case=case['id'],branch=bid,denoise_step=step+1,denoise_elapsed_seconds=sum(times))
   assert len(times)==8 and all(n>=8 for n in counts.values()) and torch.isfinite(vr).all() and torch.isfinite(ar).all()
   progress('decode',case=case['id'],branch=bid)
   video_latents=unpatchify(vr,shape);audio_latents=ar.reshape(2,af,32).permute(0,2,1);decode_and_save(video_latents,audio_latents,vae,audio_vae,str(video),'cuda:0')
   meta={'case':case['id'],'branch':bid,'title_ru':case['title_ru'],'checkpoint_step':4000,'training_architecture':branch['training_base'],'architecture':'vdn','adapter_sha256':branch['adapter_sha256'],'strength':1,'learned_style_merged':False,'style_application':'separate BF16 residual on frozen FP8 base','base_FP8_scope':info['FP8']['linears'],'style_forward_counts':dict(counts),'prompt':case['prompt'],'prompt_sha256':case['prompt_sha256'],'seed':case['seed'],'initial_noise_sha256':noise,'reference_ids':[],'reference_mode':'none','NFE':8,'frames':345,'width':case['width'],'height':case['height'],'fps':24,'video_sha256':sha256(video),'denoise_seconds':sum(times),'generation_and_decode_seconds':time.perf_counter()-started,'training_updates_performed':0,'attention_backend':'decomposed','inference_model_info':info}
   write_json(video.with_suffix('.json'),meta);results.append(meta);pair.append(meta);progress('video_completed',case=case['id'],branch=bid);print('DONE',case['id'],bid,round(meta['generation_and_decode_seconds'],2),flush=True)
   del payload,pack,vr,ar,v,audio,video_latents,audio_latents;gc.collect()
  assert len(pair)==2 and pair[0]['initial_noise_sha256']==pair[1]['initial_noise_sha256'] and pair[0]['prompt_sha256']==pair[1]['prompt_sha256']
  write_json(out/(case['id']+'_pair_verified.json'),{'verified':True,'case':case['id'],'same_initial_noise':True,'same_prompt':True,'branches':[r['branch'] for r in pair]})
 for h in hooks:h.remove()
 if a.merge_controls:
  for name,module in modules:
   owner,leaf=name.rsplit('.',1);setattr(model.get_submodule(owner),leaf,module.base)
  modules.clear();hooks.clear();del module;gc.collect();torch.cuda.empty_cache()
  control=next(c for c in suite['cases'] if c['id']=='N03');reference_result=next(r for r in results if r['case']=='N03')
  for branch in suite['branches']:
   if (root/'GLOBAL_STOP').exists():raise InterruptedError('Stop before merged control')
   progress('restore_pristine_base_and_merge',case='N03',branch=branch['id']+'_merged')
   for name,(weight,bias) in pristine.items():
    if '.' in name:owner,leaf=name.rsplit('.',1);parent=model.get_submodule(owner)
    else:leaf=name;parent=model
    layer=nn.Linear(weight.shape[1],weight.shape[0],bias=bias is not None,device='meta',dtype=weight.dtype)
    layer.weight=nn.Parameter(weight.to('cuda:0'),requires_grad=False)
    if bias is not None:layer.bias=nn.Parameter(bias.to('cuda:0'),requires_grad=False)
    setattr(parent,leaf,layer)
   with safe_open(str(adapters[branch['id']]),framework='pt',device='cpu') as f:
    keys=set(f.keys());merged=0
    for name,layer in model.named_modules():
     if not TARGET.match(name):continue
     source=name if name+'.lora_A.weight' in keys else name.replace('.attn.orig.','.attn.')
     A=f.get_tensor(source+'.lora_A.weight').float().to('cuda:0');B=f.get_tensor(source+'.lora_B.weight').float().to('cuda:0')
     layer.weight.add_((B@A).to(layer.weight.dtype));merged+=1
    del A,B
   assert merged==300
   fp8_count=convert_linear_to_fp8(model,skip_end_blocks=0);assert fp8_count==363
   payload=torch.load(cache/'N03.pt',map_location='cpu',weights_only=True);assert payload['case']==control
   t=video_latent_num_frames(345,17,5);af=audio_latent_num_frames(345);shape=(model.config.in_channels,t,48,84)
   pack=PackedCondition({'prompt_embeds':payload['prompt_embeds'],'text_token_tags':payload['text_token_tags'],'condition_latents':[]},shape,af,'cuda:0')
   gen=torch.Generator('cuda:0').manual_seed(control['seed']);vn=torch.randn((1,*shape),generator=gen,device='cuda:0');ar=torch.randn((af*2,32),generator=gen,device='cuda:0')
   noise={'video':hashlib.sha256(vn.cpu().numpy().tobytes()).hexdigest(),'audio':hashlib.sha256(ar.cpu().numpy().tobytes()).hexdigest()};assert noise==reference_result['initial_noise_sha256'];vr=patchify(vn);del vn
   vs=MiniMaxH3Scheduler(shift=12);aus=MiniMaxH3Scheduler(shift=3);vs.set_timesteps(9,device='cuda:0');aus.set_timesteps(9,device='cuda:0');times=[];started=time.perf_counter()
   progress('generate_merged',case='N03',branch=branch['id']+'_merged')
   for step,(tv,ta) in enumerate(zip(vs.timesteps,aus.timesteps)):
    if (root/'GLOBAL_STOP').exists():raise InterruptedError('Stop at merged denoise boundary')
    torch.cuda.synchronize();s=time.perf_counter();v,audio=predict(model,pack,vr,ar,float(tv),float(ta),[]);vr=vs.step(v,tv,vr,return_dict=False)[0];ar=aus.step(audio,ta,ar,return_dict=False)[0];torch.cuda.synchronize();times.append(time.perf_counter()-s)
    progress('generate_merged',case='N03',branch=branch['id']+'_merged',denoise_step=step+1)
   assert len(times)==8 and torch.isfinite(vr).all() and torch.isfinite(ar).all()
   video_latents=unpatchify(vr,shape);audio_latents=ar.reshape(2,af,32).permute(0,2,1);dest=out/'N03'/(branch['id']+'_merged');dest.mkdir(parents=True,exist_ok=True);video=dest/(f'N03_seed{control["seed"]}.mp4');decode_and_save(video_latents,audio_latents,vae,audio_vae,str(video),'cuda:0')
   meta={**reference_result,'branch':branch['id']+'_merged','training_architecture':branch['training_base'],'adapter_sha256':branch['adapter_sha256'],'learned_style_merged':True,'style_application':'merge FP32 delta into pristine BF16 base, then official FP8 requantization','style_forward_counts':{},'merged_style_pairs':300,'video_sha256':sha256(video),'denoise_seconds':sum(times),'generation_and_decode_seconds':time.perf_counter()-started}
   write_json(video.with_suffix('.json'),meta);results.append(meta);progress('video_completed',case='N03',branch=meta['branch']);print('DONE','N03',meta['branch'],round(meta['generation_and_decode_seconds'],2),flush=True)
   del payload,pack,vr,ar,v,audio,video_latents,audio_latents;gc.collect();torch.cuda.empty_cache()
 for b in suite['branches']:assert sha256(adapters[b['id']])==b['adapter_sha256']
 assert len(results)==expected_total
 write_json(out/'generation_verified.json',{'complete':True,'videos':expected_total,'all_six_pairs_same_target_noise':True,'merge_control_videos':2 if a.merge_controls else 0,'adapters_unchanged':True,'training_updates':0,'cases':results});progress('completed')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  try:
   import sys
   root=Path(sys.argv[sys.argv.index('--root')+1]);p=root/'work/generalization_20261006';p.mkdir(exist_ok=True);write_json(p/'failure.json',{'error_type':type(exc).__name__,'error':str(exc)[:250],'time_epoch':time.time()})
  except Exception:pass
  raise
