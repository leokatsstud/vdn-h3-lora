import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stylelib import GPU_ROOT
from stylelib.common import read_json,write_json,fingerprint,sha256,atomic_torch_save,code_fingerprint
from stylelib.conditioning import condition_tokens

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    from filelock import FileLock
    args.out.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.out/'.lock'),timeout=0):run(args)
def run(args):
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition,get_qwen3vl_prompt_embeds
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    from src.inference.encode_keyframes import normalize_references
    from src.inference.render import PIXEL_MEAN,PIXEL_STD
    suite=read_json(GPU_ROOT/'validation_suite.json');sources=read_json(args.weights/'sources.json');started=time.perf_counter()
    key=fingerprint({'suite':suite,'sources':sources,'refs':{p:sha256(GPU_ROOT/p) for c in suite['cases'] for p in c['references']},'ref_size':768,'preprocessor_code':code_fingerprint(['prepare_validation_cache.py','stylelib/conditioning.py'])})
    if (args.out/'index.json').exists():
        old=read_json(args.out/'index.json')
        if old['key']!=key:raise ValueError('Validation inputs changed; choose a new cache directory')
        if old['complete']:print('Validation cache already complete');return
    images={c['id']:normalize_references([Image.open(GPU_ROOT/p) for p in c['references']],768) for c in suite['cases']}
    pending=[c for c in suite['cases'] if not (args.out/(c['id']+'.refs.pt')).exists()]
    if pending:
        vae=AutoencoderKLMiniMaxH3.from_pretrained(str(args.weights/'vdn/h3-base'),subfolder='vae').to(args.device).eval().requires_grad_(False);vae.enable_tiling()
        with torch.no_grad():
            for c in pending:
                refs=[encode_vae_condition(vae,torch.from_numpy(np.array(im)).permute(2,0,1)[None,:,None].to(args.device),PIXEL_MEAN,PIXEL_STD,42) for im in images[c['id']]]
                atomic_torch_save({'key':key,'refs':refs},args.out/(c['id']+'.refs.pt'))
        del vae;gc.collect();torch.cuda.empty_cache()
    pending=[c for c in suite['cases'] if not (args.out/(c['id']+'.pt')).exists()]
    if pending:
        processor=Qwen3VLProcessor.from_pretrained(str(args.weights/'text'),subfolder='processor')
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(args.weights/'text'),subfolder='text_encoder',dtype=torch.bfloat16).to(args.device).eval().requires_grad_(False)
        with torch.no_grad():
            for c in pending:
                ids,tags,vision=condition_tokens(processor,c['prompt'],images[c['id']])
                embeds=get_qwen3vl_prompt_embeds(model,processor,ids,vision_inputs=vision,device=torch.device(args.device),dtype=torch.bfloat16)[0].cpu()
                refs=torch.load(args.out/(c['id']+'.refs.pt'),map_location='cpu',weights_only=True)
                if refs['key']!=key:raise ValueError('Stale reference cache')
                atomic_torch_save({'key':key,'case':c,'prompt_embeds':embeds,'text_token_tags':torch.tensor(tags),'condition_latents':refs['refs']},args.out/(c['id']+'.pt'));print('Cached',c['id'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(args.out/'index.json',{'key':key,'sources':sources,'complete':True,'gpu_seconds':time.perf_counter()-started,'suite':suite})
if __name__=='__main__':main()
