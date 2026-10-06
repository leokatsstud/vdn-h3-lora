"""GPU phases: visual VAE, audio VAE, Qwen3-VL. One model resident at a time."""
import argparse,gc,math,time,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stylelib.common import load_manifest,read_json,write_json,cache_key,verify_record,atomic_torch_save,contract,fingerprint,code_fingerprint
from stylelib.conditioning import condition_tokens,UNCONDITIONAL_PROBE

def decode_video(path,frames,h,w):
    raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    array=np.frombuffer(raw,dtype=np.uint8).copy().reshape(-1,h,w,3)
    if len(array)!=frames:raise ValueError('Decoded frame count differs from manifest')
    return torch.from_numpy(array).permute(3,0,1,2)[None]
def decode_audio(path,samples,rate,present):
    if not present:return torch.zeros(2,samples),0
    raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-ac','2','-ar',str(rate),'-f','f32le','pipe:1'])
    arr=np.frombuffer(raw,dtype='<f4').copy().reshape(-1,2).T;valid=min(arr.shape[1],samples)
    wave=torch.zeros(2,samples);wave[:,:valid]=torch.from_numpy(arr[:,:valid]);return wave,valid
def release(model):
    del model;gc.collect();torch.cuda.empty_cache()
def check_existing(path,key):
    if not path.exists():return False
    cached=torch.load(path,map_location='cpu',weights_only=True)
    if cached['key']!=key:raise ValueError('Cache key mismatch: '+str(path))
    return True
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['media','text','all'],default='all');ap.add_argument('--ref-size',type=int,default=768);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    from filelock import FileLock
    args.out.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.out/'.cache.lock'),timeout=0):run(args)
def run(args):
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition,get_qwen3vl_prompt_embeds
    from src.inference.encode_keyframes import normalize_references
    from src.inference.render import PIXEL_MEAN,PIXEL_STD
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    manifest,root=load_manifest(args.manifest);weights=args.weights.resolve();cfg=contract();sources=read_json(weights/'sources.json')
    if sources['vdn_revision']!=cfg['hf_revision']:raise ValueError('Wrong weight revision')
    settings={'sources':sources,'ref_size':args.ref_size,'encode_seed':42,'video_encoding':'upstream encode_vae_condition (sample+fp16 round+normalization)','audio_encoding':'posterior mode normalized; stereo channel-major; native rounded 40-Hz grid','contract_sha256':fingerprint(cfg),'unconditional_probe':'single_space','preprocessor_code':code_fingerprint(['prepare_style_cache.py','stylelib/conditioning.py'])}
    index_path=args.out/'index.json'
    old=read_json(index_path) if index_path.exists() else None
    if old and (old['settings']!=settings or old['records_sha256']!=manifest['records_sha256']):raise ValueError('Inputs/contract changed: use a NEW cache directory')
    index=old or {'schema':2,'settings':settings,'records_sha256':manifest['records_sha256'],'gpu_phase_seconds':0.,'records':[]}
    entries=[]
    for rec in manifest['records']:
        verify_record(rec,root);key=cache_key(rec,settings);stem=f'{rec["id"]}.{key[:16]}'
        modes=['t2v']+(['ref'] if rec['references'] else [])
        entries.append({'record':rec,'key':key,'media':stem+'.media.pt','visual':stem+'.visual.pt','conditions':{mode:stem+'.'+mode+'.pt' for mode in modes}})
    index['records']=entries
    started=time.perf_counter()
    try:
        if args.phase in ('media','all'):
            pending=[e for e in entries if not check_existing(args.out/e['visual'],e['key'])]
            if pending:
                vae=AutoencoderKLMiniMaxH3.from_pretrained(str(weights/'vdn/h3-base'),subfolder='vae').to(args.device).eval().requires_grad_(False)
                vae.enable_tiling()
                if vae.config.clip_length!=17 or vae.config.token_drop!=3:raise ValueError('Unexpected video VAE chunk/mask geometry')
                with torch.no_grad():
                    for e in pending:
                        r=e['record'];pixels=decode_video(root/r['video'],r['frames'],r['height'],r['width']).to(args.device)
                        latent=encode_vae_condition(vae,pixels,PIXEL_MEAN,PIXEL_STD,42)[0];del pixels
                        shape=(24,(r['frames']-5)//17*5+2,r['height']//16,r['width']//16)
                        if tuple(latent.shape)!=shape:raise ValueError(f'{r["id"]}: VAE shape {tuple(latent.shape)} != {shape}')
                        images=normalize_references([Image.open(root/x['path']) for x in r['references']],args.ref_size)
                        refs=[encode_vae_condition(vae,torch.from_numpy(np.array(im)).permute(2,0,1)[None,:,None].to(args.device),PIXEL_MEAN,PIXEL_STD,42) for im in images]
                        # Temporal encoding operates on independent 17-frame chunks. If
                        # explicit dataset padding touches a chunk, exclude ALL its rows.
                        valid_t=shape[1] if not r['artificial_tail_frames'] else min(shape[1],((r['frames']-r['artificial_tail_frames'])//17)*5)
                        mask=torch.arange(shape[1])<valid_t
                        atomic_torch_save({'key':e['key'],'video_latents':latent,'video_valid_frames':mask,'condition_latents':refs},args.out/e['visual'])
                        print(r['id'],'visual cached',shape,flush=True)
                vae.to('cpu');del vae;gc.collect();torch.cuda.empty_cache()
            pending=[e for e in entries if not check_existing(args.out/e['media'],e['key'])]
            if pending:
                audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(weights/'vdn/h3-base'),subfolder='audio_vae').to(args.device).eval().requires_grad_(False)
                rate=audio_vae.config.sampling_rate;hop=audio_vae.hop_length
                if rate/hop!=40:raise ValueError('Unexpected audio latent rate')
                with torch.no_grad():
                    for e in pending:
                        r=e['record'];media=torch.load(args.out/e['visual'],map_location='cpu',weights_only=True)
                        count=round(r['frames']/24*40);wave,n=decode_audio(root/r['video'],count*hop,rate,r['audio_present'])
                        z=audio_vae.encode(wave[:,None].to(args.device),return_dict=False)[0].mode().float().cpu()
                        mean=torch.tensor(audio_vae.config.latents_mean).view(1,-1,1);std=torch.tensor(audio_vae.config.latents_std).view(1,-1,1)
                        z=(z-mean)/std
                        if tuple(z.shape)!=(2,32,count):raise ValueError('Audio shape mismatch')
                        valid=min(n//hop,int((r['frames']-r['artificial_tail_frames'])/24*40))
                        media.update(audio_latents=z,audio_valid_rows=(torch.arange(count)<valid).repeat(2),audio_supervised=r['audio_supervised'],audio_samples_valid=n)
                        atomic_torch_save(media,args.out/e['media']);print(r['id'],'audio cached',count,flush=True)
                audio_vae.to('cpu');del audio_vae;gc.collect();torch.cuda.empty_cache()
        if args.phase in ('text','all'):
            jobs=[(e,mode) for e in entries for mode,path in e['conditions'].items() if not check_existing(args.out/path,e['key'])]
            if jobs:
                processor=Qwen3VLProcessor.from_pretrained(str(weights/'text'),subfolder='processor')
                encoder=Qwen3VLForConditionalGeneration.from_pretrained(str(weights/'text'),subfolder='text_encoder',dtype=torch.bfloat16).to(args.device).eval().requires_grad_(False)
                text_uncond=None
                for e,mode in jobs:
                    r=e['record'];images=normalize_references([Image.open(root/x['path']) for x in r['references']],args.ref_size) if mode=='ref' else []
                    prompt=(root/(r['reference_caption'] if mode=='ref' else r['caption'])).read_text(encoding='utf-8-sig').strip()
                    def encode(prompt):
                        ids,tags,vision=condition_tokens(processor,prompt,images)
                        limit=getattr(encoder.config.text_config,'max_position_embeddings',None)
                        if limit and len(ids)>limit:raise ValueError('Conditioning exceeds encoder context; do not truncate captions')
                        with torch.no_grad():embedding=get_qwen3vl_prompt_embeds(encoder,processor,ids,vision_inputs=vision,text_encoder_layer=50,device=torch.device(args.device),dtype=torch.bfloat16)[0].cpu()
                        return {'prompt_embeds':embedding,'text_token_tags':torch.tensor(tags),'prompt':prompt,'token_count':len(ids)}
                    cond=encode(prompt)
                    if mode=='t2v' and text_uncond is not None:uncond=text_uncond
                    else:
                        uncond=encode(UNCONDITIONAL_PROBE)
                        if mode=='t2v':text_uncond=uncond
                    atomic_torch_save({'key':e['key'],'mode':mode,'conditional':cond,'unconditional':uncond},args.out/e['conditions'][mode]);print(r['id'],mode,'text cached',cond['token_count'],flush=True)
                encoder.to('cpu');del encoder;gc.collect();torch.cuda.empty_cache()
    finally:
        index['gpu_phase_seconds']+=time.perf_counter()-started
        index['complete']=all((args.out/e['media']).exists() and all((args.out/p).exists() for p in e['conditions'].values()) for e in entries)
        write_json(index_path,index)
    print('Cache complete:',index['complete'],flush=True)
if __name__=='__main__':main()
