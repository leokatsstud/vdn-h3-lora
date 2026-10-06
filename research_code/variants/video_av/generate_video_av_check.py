"""Official tuned FP8 assembly/sampler, external style LoRA merged before quantization."""
import argparse,gc,json,sys,time,zipfile
from pathlib import Path
import torch
from stylelib.common import read_json,write_json,sha256
from stylelib import GPU_ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();job=read_json(a.config)
    root=Path(job['root']);work=root/'work';private=root/'private';work.mkdir(parents=True,exist_ok=True);deadline=job['deadline_epoch']
    def state(stage,status='running',**extra):write_json(work/'pipeline_state.json',dict(status=status,stage=stage,phase='video_av_check',deadline_epoch=deadline,**extra))
    try:
        state('prepare_reference_and_text')
        import subprocess
        cached=read_json(work/'paired_cache/index.json');requested=read_json(job['suite'])
        if cached['suite']['cases']!=requested['cases'] or cached['reference_sha256']!=requested['reference_image_sha256']:raise ValueError('Reused V110 conditioning differs')
        from omegaconf import OmegaConf
        from src.config.inference import InferenceConfig
        from src.inference.utils.assemble import build_inference_model,render_record
        from src.inference.render import generate_latents,decode_and_save
        suite=read_json(job['suite']);index=read_json(work/'paired_cache/index.json');checks=[]
        for point in suite['checkpoints']:
            if sha256(point['adapter'])!=point['adapter_sha256']:raise ValueError('Selected adapter changed')
            state('load_official_fast_model',checkpoint=point['step'])
            cfg=OmegaConf.merge(OmegaConf.structured(InferenceConfig),OmegaConf.load(str(GPU_ROOT/'upstream/configs/inference/8nfe_tuned_fp8.yaml')),
                {'checkpoint':str(Path(job['weights'])/'vdn/stage-dmd-step-250'),'base_source':str(Path(job['weights'])/'vdn/h3-base'),'vae_source':str(Path(job['weights'])/'vdn/h3-base'),
                 'external_loras':[{'path':point['adapter'],'alpha':32.0}], 'render':{'num_frames':345,'num_steps':8,'warmup_steps':2,'seed':20261003,'device':'cuda:0','record':True,'prompt_file':'cached','out':'pending'}})
            torch.set_grad_enabled(False);model=build_inference_model(cfg,'cuda:0')
            if not model.fp8_linears or model.softmax_backend!='decomposed' or model.external_loras[0]['pairs']!=300 or model.external_loras[0]['scale']!=1:raise ValueError('Official fast runtime or LoRA scope not active')
            for i,case in enumerate(suite['cases']):
                condition=torch.load(work/'paired_cache'/(case['id']+'.pt'),map_location='cpu',weights_only=True)
                embeds=condition['prompt_embeds'].to('cuda:0',torch.bfloat16);tags=condition['text_token_tags']
                refs=condition['condition_latents'];conditions=(tuple('ref' for _ in refs),[z.to('cuda:0',torch.float32) for z in refs]) if refs else None
                if i==0:
                    state('warmup_fast_kernels',checkpoint=point['step']);generate_latents(model.transformer,embeds,tags,345,2,20261003,'cuda:0',conditions=conditions)
                state('fast_generate',checkpoint=point['step'],case=case['id']);times=[]
                v,au=generate_latents(model.transformer,embeds,tags,345,8,20261003,'cuda:0',step_seconds=times,conditions=conditions)
                dest=work/'videos'/f'step{point["step"]}';dest.mkdir(parents=True,exist_ok=True);video=dest/(case['id']+'_seed20261003.mp4')
                decode_and_save(v,au,model.vae,model.audio_vae,str(video),'cuda:0')
                runtime=render_record(cfg,model);runtime['timings']={'step_seconds':times,'denoise_seconds':sum(times)}
                write_json(video.with_suffix('.inference.json'),runtime)
                meta={'checkpoint_step':point['step'],'case':case['id'],'prompt':case['prompt'],'seed':20261003,'frames':345,'width':1344,'height':768,'strength':1,'adapter_sha256':point['adapter_sha256'],'reference_image_sha256':suite['reference_image_sha256'] if refs else None,'profile':'official8nfe_tuned_fp8','fp8_linears':model.fp8_linears,'softmax_backend':model.softmax_backend,'inference_kernels':True,'denoise_seconds':sum(times),'video_sha256':sha256(video),'training_updates_performed':0,'native_reference_rng_order':True}
                write_json(video.with_suffix('.json'),meta);checks.append(meta);print('DONE',point['step'],case['id'],'denoise_seconds',sum(times),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        archive=work/job['archive_name']
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in (work/'videos').rglob('*'):
                if f.is_file():z.write(f,'videos/'+f.relative_to(work/'videos').as_posix())
            z.write(Path(job['suite']),'suite.json')
        write_json(work/'generation_verified.json',{'videos':len(checks),'checkpoints':[p['step'] for p in suite['checkpoints']],'profile':'official8nfe_tuned_fp8','training_updates_performed':0})
        write_json(work/'artifacts.json',{'step':suite['checkpoints'][-1]['step'],'files':[{'name':archive.name,'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),'download_to_laptop':True}]})
        state('generated_waiting_manual_verification','completed_generated',actual_step=suite['checkpoints'][-1]['step'])
    except Exception as exc:state('failed','failed',error_type=type(exc).__name__);raise

if __name__=='__main__':main()
