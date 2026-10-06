"""Five LoRA-only videos. No training, upload/download or rental operations.
Original H3: one124-frame clip. VDN: two345-frame T2V/reference pairs.
Target noise is identical within each VDN pair; reference noise uses its own RNG.
"""
import argparse,time,gc,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
from stylelib.common import read_json,write_json,contract,sha256
from stylelib.image_recipe import apply_image_recipe
from stylelib.transfer_model import build_style_model
from stylelib.adapter import style_modules
from stylelib.packing import PackedCondition,patchify,unpatchify,predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root;work=root/'work';weights=work/'weights'
    suite=read_json(root/'private/4000_final_suite.json');cache=work/'4000_final_cache';index=read_json(cache/'index.json');assert index['complete'] and index['suite']==suite and suite['checkpoint_step']==4000
    progress=read_json(work/'runs/dense_fp8_images5000/progress.json');assert progress['step']==4000 and progress['status']=='stopped'
    adapter=Path(suite['adapter']);expected=suite['adapter_sha256'];assert sha256(adapter)==expected
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio,MiniMaxH3Scheduler
    from src.inference.render import decode_and_save
    from src.models.hybrid_transform import set_inference_mode
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames,audio_latent_num_frames
    torch.set_grad_enabled(False);results=[];ref_seed=21261008
    for arch in ['dense','vdn']:
        cases=[c for c in suite['cases'] if c['architecture']==arch];nfe=50 if arch=='dense' else 8
        ready=[]
        for case in cases:
            video=work/'4000_final_videos'/case['id']/(case['id']+f'_seed{suite["seed"]}.mp4')
            if video.exists() and video.with_suffix('.json').exists():
                meta=read_json(video.with_suffix('.json'));assert sha256(video)==meta['video_sha256'] and meta['adapter_sha256']==expected and meta['prompt']==case['prompt'] and meta['frames']==case['frames'];ready.append(meta)
        if len(ready)==len(cases):results.extend(ready);continue
        write_json(work/'4000_generation_progress.json',{'stage':'load_model','architecture':arch,'completed':len(results)})
        cfg=apply_image_recipe(contract(),'full_standard')
        if arch=='dense':cfg['active_frozen_adapters']=[];cfg['style']['target_suffixes']=[s.replace('attn.orig.','attn.') for s in cfg['style']['target_suffixes']]
        model,info=build_style_model(weights,cfg,'cuda:0',False,'decomposed' if arch=='vdn' else 'flex',arch,'fp8');modules=style_modules(model);assert len(modules)==300
        with safe_open(str(adapter),framework='pt',device='cpu') as f:
            assert len(f.keys())==600
            for name,m in modules:
                source=name.replace('.attn.orig.','.attn.');A=f.get_tensor(source+'.lora_A.weight');B=f.get_tensor(source+'.lora_B.weight')
                assert A.shape==m.style_A.shape and B.shape==m.style_B.shape;m.style_A.copy_(A.to('cuda:0',torch.float32));m.style_B.copy_(B.to('cuda:0',torch.float32));m.strength=1
        model.requires_grad_(False)
        if arch=='vdn':set_inference_mode(model,True)
        counts={n:0 for n,_ in modules};hooks=[]
        for name,m in modules:
            def hook(module,inputs,output,key=name):counts[key]+=1
            hooks.append(m.register_forward_hook(hook))
        vae=AutoencoderKLMiniMaxH3.from_pretrained(str(weights/'vdn/h3-base'),subfolder='vae').to('cuda:0').eval().requires_grad_(False);vae.enable_tiling()
        audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(weights/'vdn/h3-base'),subfolder='audio_vae').to('cuda:0').eval().requires_grad_(False)
        for case in cases:
            existing=next((r for r in ready if r['case']==case['id']),None)
            if existing:results.append(existing);continue
            if (root/'GLOBAL_STOP').exists():raise InterruptedError('Explicit stop before generation')
            payload=torch.load(cache/(case['id']+'.pt'),map_location='cpu',weights_only=True);assert payload['key']==index['key'] and payload['case']==case
            t=video_latent_num_frames(case['frames'],17,5);af=audio_latent_num_frames(case['frames']);shape=(model.config.in_channels,t,case['height']//16,case['width']//16)
            condition={'prompt_embeds':payload['prompt_embeds'],'text_token_tags':payload['text_token_tags'],'condition_latents':payload['condition_latents']};pack=PackedCondition(condition,shape,af,'cuda:0')
            rg=torch.Generator('cuda:0').manual_seed(ref_seed);refs=[patchify(.999*z+.001*torch.randn(z.shape,generator=rg,device='cuda:0')) for z in pack.refs]
            vg=torch.Generator('cuda:0').manual_seed(suite['seed']);vn=torch.randn((1,*shape),generator=vg,device='cuda:0');ar=torch.randn((af*2,32),generator=vg,device='cuda:0');noise={'video':hashlib.sha256(vn.cpu().numpy().tobytes()).hexdigest(),'audio':hashlib.sha256(ar.cpu().numpy().tobytes()).hexdigest()};vr=patchify(vn);del vn
            vs=MiniMaxH3Scheduler(shift=12);aus=MiniMaxH3Scheduler(shift=3);vs.set_timesteps(nfe+1,device='cuda:0');aus.set_timesteps(nfe+1,device='cuda:0');times=[]
            for n in counts:counts[n]=0
            write_json(work/'4000_generation_progress.json',{'stage':'generate','architecture':arch,'case':case['id'],'completed':len(results),'frames':case['frames'],'NFE':nfe})
            for i,(tv,ta) in enumerate(zip(vs.timesteps,aus.timesteps)):
                if (root/'GLOBAL_STOP').exists():raise InterruptedError('Explicit stop at denoise-step boundary')
                torch.cuda.synchronize();start=time.perf_counter();v,a=predict(model,pack,vr,ar,float(tv),float(ta),refs);vr=vs.step(v,tv,vr,return_dict=False)[0];ar=aus.step(a,ta,ar,return_dict=False)[0];torch.cuda.synchronize();times.append(time.perf_counter()-start)
                if i%10==0:print(case['id'],'evaluation',i,flush=True)
            assert len(times)==nfe and all(n>=nfe for n in counts.values()) and torch.isfinite(vr).all() and torch.isfinite(ar).all()
            video_latents=unpatchify(vr,shape);audio_latents=ar.reshape(2,af,32).permute(0,2,1);dest=work/'4000_final_videos'/case['id'];dest.mkdir(parents=True,exist_ok=True);video=dest/(case['id']+f'_seed{suite["seed"]}.mp4');decode_and_save(video_latents,audio_latents,vae,audio_vae,str(video),'cuda:0')
            meta={'case':case['id'],'variant':'step4000_'+case['id'],'architecture':arch,'source_video':case['source_video'],'checkpoint_step':4000,'adapter_sha256':expected,'strength':1,'learned_style_merged':False,'style_application':'separate BF16 residual on frozen FP8 base','base_FP8_scope':info['FP8']['linears'],'style_forward_counts':dict(counts),'prompt':case['prompt'],'original_caption_sha256':case['original_caption_sha256'],'seed':suite['seed'],'reference_noise_seed':ref_seed,'initial_noise_sha256':noise,'reference_image_sha256':[suite['references'][rid]['sha256'] for rid in case['references']],'reference_ids':case['references'],'reference_mode':'appearance ref2va-like, not fixed first frame' if case['references'] else 'none','NFE':nfe,'frames':case['frames'],'width':case['width'],'height':case['height'],'fps':24,'video_sha256':sha256(video),'denoise_seconds':sum(times),'training_updates_performed':0}
            write_json(video.with_suffix('.json'),meta);write_json(video.with_suffix('.inference.json'),{'model':info,'inference_kernels':arch=='vdn','softmax_backend':'decomposed' if arch=='vdn' else 'dense','step_seconds':times,'style_application':meta['style_application']});results.append(meta);print('DONE',case['id'],sum(times),flush=True)
            del payload,pack,refs,vr,ar,v,a,video_latents,audio_latents
        for h in hooks:h.remove()
        del model,vae,audio_vae,modules,hooks;gc.collect();torch.cuda.empty_cache()
    assert len(results)==5 and sha256(adapter)==expected
    for vid in ['V069','V073']:
        pair=[r for r in results if r['architecture']=='vdn' and r['source_video']==vid];assert len(pair)==2 and pair[0]['initial_noise_sha256']==pair[1]['initial_noise_sha256']
    write_json(work/'4000_generation_verified.json',{'complete':True,'videos':5,'cases':results,'no_more_training':True,'no_baselines':True,'adapter_unchanged':True});write_json(work/'4000_generation_progress.json',{'stage':'completed','completed':5})
if __name__=='__main__':main()
