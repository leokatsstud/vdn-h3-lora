"""Six requested official native-fast renders only. No train/upload/download/delete."""
import argparse,gc,json,hashlib,zipfile
from pathlib import Path
import torch
from stylelib.common import read_json,write_json,sha256
from stylelib import GPU_ROOT
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);job=read_json(p.parse_args().config);work=Path(job['root'])/'work';suite=read_json(job['suite']);index=read_json(work/'final_cache/index.json');assert index['complete'] and index['cases']==suite['cases'] and index['references']==suite['references'];cases={x['id']:x for x in suite['cases']};checks=[]
    from omegaconf import OmegaConf
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model,render_record
    import src.inference.render as render
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames,audio_latent_num_frames
    def state(stage,status='running',**extra):write_json(work/'pipeline_state.json',dict(stage=stage,status=status,phase='final_six',**extra))
    def noise_hashes(model,case,refs):
        generator=torch.Generator('cuda:0').manual_seed(suite['seed'])
        for ref in refs:torch.randn(ref.shape,device='cuda:0',generator=generator,dtype=torch.float32)
        v=torch.randn((1,model.transformer.config.in_channels,video_latent_num_frames(case['frames'],17,5),render.LATENT_H,render.LATENT_W),device='cuda:0',generator=generator,dtype=torch.float32)
        au=torch.randn((audio_latent_num_frames(case['frames'])*2,32),device='cuda:0',generator=generator,dtype=torch.float32)
        return hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest(),hashlib.sha256(au.cpu().numpy().tobytes()).hexdigest()
    try:
        for point in suite['checkpoints']:
            assert sha256(point['adapter'])==point['adapter_sha256'];state('load_native_fast_model',checkpoint=point['label'])
            cfg=OmegaConf.merge(OmegaConf.structured(InferenceConfig),OmegaConf.load(str(GPU_ROOT/'upstream/configs/inference/8nfe_tuned_fp8.yaml')),{'checkpoint':str(Path(job['weights'])/'vdn/stage-dmd-step-250'),'base_source':str(Path(job['weights'])/'vdn/h3-base'),'vae_source':str(Path(job['weights'])/'vdn/h3-base'),'external_loras':[{'path':point['adapter'],'alpha':32.0*point['strength']}],'render':{'num_frames':243,'num_steps':8,'warmup_steps':2,'seed':suite['seed'],'device':'cuda:0','record':True,'prompt_file':'cached','out':'pending'}})
            torch.set_grad_enabled(False);model=build_inference_model(cfg,'cuda:0');assert model.fp8_linears==363 and model.softmax_backend=='decomposed' and model.external_loras[0]['pairs']==300 and model.external_loras[0]['scale']==point['strength']
            warmed=set()
            for case_id in point['cases']:
                case=cases[case_id];payload=torch.load(work/'final_cache'/(case_id+'.pt'),map_location='cpu',weights_only=True);assert payload['key']==index['key'] and payload['case']==case
                render.LATENT_H,render.LATENT_W=case['height']//16,case['width']//16
                embeds=payload['prompt_embeds'].to('cuda:0',torch.bfloat16);tags=payload['text_token_tags'];refs=[x.to('cuda:0',torch.float32) for x in payload['condition_latents']];conditions=(tuple('ref' for _ in refs),refs) if refs else None
                shape=(render.LATENT_H,render.LATENT_W)
                if shape not in warmed:state('warmup_shape',checkpoint=point['label'],case=case_id);render.generate_latents(model.transformer,embeds,tags,case['frames'],2,suite['seed'],'cuda:0',conditions=conditions);warmed.add(shape)
                vh,ah=noise_hashes(model,case,refs);state('generate',checkpoint=point['label'],case=case_id,completed=len(checks));times=[]
                v,au=render.generate_latents(model.transformer,embeds,tags,case['frames'],8,suite['seed'],'cuda:0',step_seconds=times,conditions=conditions)
                assert torch.isfinite(v).all() and torch.isfinite(au).all();dest=work/'videos'/point['label'];dest.mkdir(parents=True,exist_ok=True);video=dest/(case_id+f'_seed{suite["seed"]}.mp4');render.decode_and_save(v,au,model.vae,model.audio_vae,str(video),'cuda:0')
                record=render_record(cfg,model);record['actual_geometry']={'width':case['width'],'height':case['height'],'frames':case['frames']};record['timings']={'step_seconds':times,'denoise_seconds':sum(times)};write_json(video.with_suffix('.inference.json'),record)
                meta={'checkpoint_label':point['label'],'checkpoint_step':point['step'],'training_phase':point['training_phase'],'case':case_id,'prompt':case['prompt'],'seed':suite['seed'],'strength':point['strength'],'width':case['width'],'height':case['height'],'frames':case['frames'],'reference_image_sha256':suite['references'][case['references'][0]]['sha256'] if refs else None,'adapter_sha256':point['adapter_sha256'],'video_sha256':sha256(video),'initial_video_noise_sha256':vh,'initial_audio_noise_sha256':ah,'denoise_seconds':sum(times),'native_fast_FP8':True,'fp8_linears':363,'backend':'decomposed','training_updates_performed':0};write_json(video.with_suffix('.json'),meta);checks.append(meta);print('DONE',point['label'],case_id,sum(times),flush=True)
                del v,au
            del model;gc.collect();torch.cuda.empty_cache()
        assert len(checks)==6
        assert len({x['initial_video_noise_sha256'] for x in checks})==1 and len({x['initial_audio_noise_sha256'] for x in checks})==1
        archive=work/'strength_sweep.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in (work/'videos').rglob('*'):
                if path.is_file():z.write(path,'videos/'+path.relative_to(work/'videos').as_posix())
            z.write(Path(job['suite']),'suite.json')
        write_json(work/'artifacts.json',{'step':60,'files':[{'name':archive.name,'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),'download_to_laptop':True}]});write_json(work/'final_six_verified.json',{'six_videos_completed':True,'matched_horizontal_noise':True,'native_portrait_geometry':True,'training_updates_performed':0});state('generated_waiting_manual_backup','completed_generated',completed=6)
    except Exception as exc:state('failed','failed',error_type=type(exc).__name__);raise
if __name__=='__main__':main()
