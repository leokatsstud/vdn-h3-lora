"""Load frozen VDN once, evaluate all milestones with/without a single reference."""
import argparse,gc,hashlib,time
from pathlib import Path
import torch
from safetensors import safe_open
from stylelib.common import read_json,write_json,sha256,contract,atomic_torch_save,fingerprint
from stylelib.model import build_style_model
from stylelib.adapter import load_style_state,style_strength
from stylelib.packing import PackedCondition,patchify,unpatchify,predict

@torch.no_grad()
def generate(model,condition,case,seed):
    from src.training.t2va_batch import few_step_sigmas,euler_blend
    frames=case['frames'];shape=(24,(frames-5)//17*5+2,case['height']//16,case['width']//16);af=round(frames/24*40)
    pack=PackedCondition(condition,shape,af,'cuda:0');rng=torch.Generator(device='cuda:0').manual_seed(seed)
    ref_rng=torch.Generator(device='cuda:0').manual_seed(seed+1000003)
    refs=[patchify(.999*z+.001*torch.randn(z.shape,device='cuda:0',generator=ref_rng,dtype=torch.float32)) for z in pack.refs]
    vr=patchify(torch.randn((1,*shape),device='cuda:0',generator=rng,dtype=torch.float32));ar=torch.randn((af*2,32),device='cuda:0',generator=rng,dtype=torch.float32)
    initial=hashlib.sha256(vr.cpu().numpy().tobytes()+ar.cpu().numpy().tobytes()).hexdigest()
    sv,sa=[x.to('cuda:0') for x in few_step_sigmas(8,12,3)];times=[]
    for i in range(8):
        torch.cuda.synchronize();start=time.perf_counter();tv=float(1-sv[i]);ta=float(1-sa[i])
        v,a=predict(model,pack,vr,ar,tv,ta,refs);vr=euler_blend(vr,v,tv,sv[i],sv[i+1]);ar=euler_blend(ar,a,ta,sa[i],sa[i+1])
        if not torch.isfinite(vr).all() or not torch.isfinite(ar).all():raise FloatingPointError('Nonfinite generation')
        torch.cuda.synchronize();times.append(time.perf_counter()-start)
    return unpatchify(vr,shape).cpu(),ar.reshape(2,af,32).permute(0,2,1).cpu(),times,initial

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
    work=a.root/'work';index=read_json(work/'paired_cache/index.json');plan=read_json(work/'comparison_plan.json');cfg=contract()
    points=[{'step':0,'adapter':None,'adapter_sha256':None},*plan['checkpoints']]
    model,info=build_style_model(a.weights,cfg,'cuda:0',False,'flex');model.eval().requires_grad_(False)
    jobs=[]
    for point in points:
        strength=0 if point['step']==0 else 1
        if point['adapter']:
            if sha256(point['adapter'])!=point['adapter_sha256']:raise ValueError('Milestone adapter changed')
            with safe_open(point['adapter'],framework='pt') as f:
                meta=f.metadata()
                if int(meta['step'])!=point['step'] or meta['vdn_revision']!=cfg['hf_revision']:raise ValueError('Adapter milestone metadata differs')
                state={k.replace('.lora_A.weight','.style_A').replace('.lora_B.weight','.style_B'):f.get_tensor(k) for k in f.keys()}
            load_style_state(model,state);del state
        for case in index['suite']['cases']:
            dest=work/'paired_tests'/f'step{point["step"]}';dest.mkdir(parents=True,exist_ok=True);path=dest/f'{case["id"]}_seed20261003.latents.pt'
            condition=torch.load(work/'paired_cache'/(case['id']+'.pt'),map_location='cpu',weights_only=True)
            identity=fingerprint({'cache_key':index['key'],'step':point['step'],'adapter_sha256':point['adapter_sha256'],'seed':20261003,'strength':strength,'reference_noise_seed':21261006})
            with style_strength(model,strength):v,au,times,initial=generate(model,condition,case,20261003)
            atomic_torch_save({'identity':identity,'video_latents':v,'audio_latents':au,'step_seconds':times,'initial_target_noise_sha256':initial},path)
            jobs.append({'path':str(path),'case':case,'step':point['step'],'strength':strength,'identity':identity,'adapter_sha256':point['adapter_sha256'],'initial_target_noise_sha256':initial})
            write_json(work/'paired_generation_progress.json',{'stage':'denoise','completed':len(jobs),'total':len(points)*2,'step':point['step'],'case':case['id']})
            print('denoised',point['step'],case['id'],sum(times),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    from src.inference.render import decode_and_save
    vae=AutoencoderKLMiniMaxH3.from_pretrained(str(a.weights/'vdn/h3-base'),subfolder='vae').to('cuda:0').eval();vae.enable_tiling()
    audio=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(a.weights/'vdn/h3-base'),subfolder='audio_vae').to('cuda:0').eval()
    with torch.no_grad():
        for i,job in enumerate(jobs,1):
            path=Path(job['path']);data=torch.load(path,map_location='cpu',weights_only=True);video=path.with_suffix('.mp4')
            decode_and_save(latents=data['video_latents'].to('cuda:0'),audio_latents=data['audio_latents'].to('cuda:0'),vae=vae,audio_vae=audio,out_path=str(video),device='cuda:0')
            write_json(video.with_suffix('.json'),{'step':job['step'],'case':job['case']['id'],'prompt':job['case']['prompt'],'seed':20261003,'strength':job['strength'],'adapter_sha256':job['adapter_sha256'],'video_sha256':sha256(video),'initial_target_noise_sha256':job['initial_target_noise_sha256'],'reference_image_sha256':index['reference_sha256'] if job['case']['references'] else None,'reference_mode':'appearance Ref2VA-like' if job['case']['references'] else 'T2V','nfe':8,'source_video':'V145','seen_dataset_probe':True})
            write_json(work/'paired_generation_progress.json',{'stage':'decode','completed':i,'total':len(jobs)})
    if len({x['initial_target_noise_sha256'] for x in jobs})!=1:raise ValueError('Target noise differs between paired conditions')
    write_json(work/'paired_tests_verified.json',{'videos':len(jobs),'checkpoints':[x['step'] for x in points],'same_target_noise_for_all':True,'reference_sha256':index['reference_sha256'],'strength_lora':1,'strength_base':0})

if __name__=='__main__':main()
