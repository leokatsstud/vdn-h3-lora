"""Native 8-NFE generation on explicit landscape/portrait shapes; decode afterward."""
import argparse,gc,time
from pathlib import Path
import torch
from stylelib.common import read_json,write_json,contract,atomic_torch_save,sha256,fingerprint
from stylelib.model import build_style_model
from stylelib.adapter import load_style_state,style_strength,merge_style
from stylelib.packing import PackedCondition,patchify,unpatchify,predict

@torch.no_grad()
def generate(model,condition,case,seed,device):
    from src.training.t2va_batch import few_step_sigmas,euler_blend
    frames=case['frames'];shape=(24,(frames-5)//17*5+2,case['height']//16,case['width']//16);af=round(frames/24*40)
    p=PackedCondition(condition,shape,af,device);rng=torch.Generator(device=device).manual_seed(seed)
    refs=[patchify(.999*z+.001*torch.randn(z.shape,device=device,generator=rng,dtype=torch.float32)) for z in p.refs]
    vr=patchify(torch.randn((1,*shape),device=device,generator=rng,dtype=torch.float32));ar=torch.randn((af*2,32),device=device,generator=rng,dtype=torch.float32)
    sv,sa=[t.to(device) for t in few_step_sigmas(8,12,3)];timings=[]
    for i in range(8):
        torch.cuda.synchronize();t=time.perf_counter();tv=float(1-sv[i]);ta=float(1-sa[i])
        v,a=predict(model,p,vr,ar,tv,ta,refs);vr=euler_blend(vr,v,tv,sv[i],sv[i+1]);ar=euler_blend(ar,a,ta,sa[i],sa[i+1])
        if not torch.isfinite(vr).all() or not torch.isfinite(ar).all():raise FloatingPointError('Nonfinite denoising state')
        torch.cuda.synchronize();timings.append(time.perf_counter()-t)
    return unpatchify(vr,shape).cpu(),ar.reshape(2,af,32).permute(0,2,1).cpu(),timings

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--adapter',type=Path);ap.add_argument('--strength',type=float,default=0);ap.add_argument('--precision',choices=['bf16','fp8'],default='bf16')
    ap.add_argument('--cases',nargs='+');ap.add_argument('--seeds',nargs='+',type=int,default=[1234,5678]);ap.add_argument('--device',default='cuda:0');ap.add_argument('--latents-only',action='store_true')
    ap.add_argument('--transformer-cpu-offload',action='store_true',help='Low-VRAM video validation with entire blocks preloaded')
    args=ap.parse_args()
    if args.strength and args.adapter is None:ap.error('Nonzero strength requires a style adapter')
    from filelock import FileLock
    args.out.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.out/'.lock'),timeout=0):run(args)
def run(args):
    cfg=contract();index=read_json(args.cache/'index.json')
    if not index['complete'] or index['sources']!=read_json(args.weights/'sources.json'):raise ValueError('Validation cache/weights mismatch')
    cases=[c for c in index['suite']['cases'] if not args.cases or c['id'] in args.cases]
    if args.cases and set(args.cases)!={c['id'] for c in cases}:raise ValueError('Unknown validation case')
    identity=fingerprint({'cache':index['key'],'strength':args.strength,'precision':args.precision,'adapter':sha256(args.adapter) if args.adapter else None,'transformer_cpu_offload':args.transformer_cpu_offload})
    if (args.out/'run.json').exists() and read_json(args.out/'run.json')['identity']!=identity:raise ValueError('Output folder belongs to another validation setting')
    jobs=[(c,s,args.out/f'{c["id"]}_seed{s}.latents.pt') for c in cases for s in args.seeds]
    pending=[j for j in jobs if not j[2].exists()]
    if pending:
        model,info=build_style_model(args.weights,cfg,'cpu' if args.transformer_cpu_offload else args.device,False,'flex');model.eval().requires_grad_(False)
        if args.adapter:
            from safetensors import safe_open
            state={}
            with safe_open(args.adapter,framework='pt') as f:
                meta=f.metadata() or {}
                if meta.get('vdn_revision')!=cfg['hf_revision'] or float(meta.get('alpha',0))!=cfg['style']['alpha']:raise ValueError('Wrong adapter base/alpha')
                for k in f.keys():state[k.replace('.lora_A.weight','.style_A').replace('.lora_B.weight','.style_B')]=f.get_tensor(k)
            load_style_state(model,state);del state
        if args.transformer_cpu_offload:
            if args.precision!='bf16':raise ValueError('CPU-offloaded validation is BF16 only')
            from accelerate import cpu_offload
            preload=sorted({type(block).__name__ for block in model.transformer_blocks})
            cpu_offload(model,execution_device=torch.device(args.device),offload_buffers=True,preload_module_classes=preload)
        if args.precision=='fp8':
            from src.models.ops.fp8_linear import convert_linear_to_fp8
            merge_style(model,args.strength);info['fp8_linears']=convert_linear_to_fp8(model,skip_end_blocks=4)
        write_json(args.out/'run.json',{'identity':identity,'model':info,'precision':args.precision,'strength':args.strength,'nfe':8,'inference_kernels':False,'purpose':'matched-seed canonical baseline/style comparison'})
        for case,seed,path in pending:
            cond=torch.load(args.cache/(case['id']+'.pt'),map_location='cpu',weights_only=True)
            if cond['key']!=index['key']:raise ValueError('Stale validation condition')
            with style_strength(model,args.strength):v,a,times=generate(model,cond,case,seed,args.device)
            atomic_torch_save({'identity':identity,'video_latents':v,'audio_latents':a,'case':case,'seed':seed,'step_seconds':times},path)
            print(case['id'],seed,'denoised',sum(times),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    if args.latents_only:return
    decode_jobs=[j for j in jobs if not j[2].with_suffix('.mp4').exists()]
    if decode_jobs:
        from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
        from src.inference.render import decode_and_save
        vae=AutoencoderKLMiniMaxH3.from_pretrained(str(args.weights/'vdn/h3-base'),subfolder='vae').to(args.device).eval();vae.enable_tiling()
        audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(args.weights/'vdn/h3-base'),subfolder='audio_vae').to(args.device).eval()
        with torch.no_grad():
            for case,seed,path in decode_jobs:
                data=torch.load(path,map_location='cpu',weights_only=True)
                if data['identity']!=identity:raise ValueError('Latent file belongs to another validation')
                decode_and_save(latents=data['video_latents'].to(args.device),audio_latents=data['audio_latents'].to(args.device),vae=vae,audio_vae=audio_vae,out_path=str(path.with_suffix('.mp4')),device=args.device)
                write_json(path.with_suffix('.json'),{'case':case['id'],'prompt':case['prompt'],'seed':seed,'identity':identity,'strength':args.strength,'precision':args.precision,'adapter_sha256':sha256(args.adapter) if args.adapter else None,'latent_sha256':sha256(path),'video_sha256':sha256(path.with_suffix('.mp4')),'step_seconds':data['step_seconds']})
if __name__=='__main__':main()
