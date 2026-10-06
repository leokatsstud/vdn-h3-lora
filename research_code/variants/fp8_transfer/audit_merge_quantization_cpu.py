"""Read-only CPU check of selected adapter deltas before/after weight requantization."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
from stylelib.common import read_json,write_json
torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
folder=root/'work/weights/vdn/h3-base/transformer';mapping=read_json(folder/'diffusion_pytorch_model.safetensors.index.json')['weight_map']
adapter=root/'work/runs/dense_fp8_images1000/style_1000.safetensors';rows=[]
def q(w):
    scale=(w.abs().amax(dim=1,keepdim=True)/448).clamp_min(1e-12)
    return (w/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
with safe_open(str(adapter),framework='pt',device='cpu') as a:
    for prefix in ['transformer_blocks.0.attn.to_q','transformer_blocks.0.ff.net.2','transformer_blocks.49.attn.to_v']:
        ak=next(k for k in a.keys() if k.startswith(prefix+'.') and 'lora_A' in k);bk=ak.replace('lora_A','lora_B')
        A=a.get_tensor(ak).float();B=a.get_tensor(bk).float();delta=B@A
        key=prefix+'.weight'
        with safe_open(str(folder/mapping[key]),framework='pt',device='cpu') as base:w=base.get_tensor(key).float()
        # Match native merge_lora_state: cast delta to the base dtype, then add
        # into BF16 weights before FP8 conversion. Do not assume an FP32 merge.
        merged=(w.to(torch.bfloat16)+delta.to(torch.bfloat16)).float()
        effective=q(merged)-q(w);dn=delta.norm();en=effective.norm()
        rows.append({'layer':prefix,'delta_to_base_relative_L2':float(dn/w.norm()),'requantized_delta_to_original_delta_norm':float(en/dn),'requantized_delta_cosine':float((effective*delta).sum()/(en*dn)),'requantized_delta_relative_error':float((effective-delta).norm()/dn),'fraction_nonzero_effective_weights':float((effective!=0).float().mean())})
        del A,B,delta,w,effective
write_json(root/'work/merge_quantization_cpu_audit.json',{'selected_layers':rows,'weight_only_check':True,'no_forward_or_quality_claim':True,'training_style_branch_BF16_inference_merged_then_FP8':True})
print(json.dumps(rows))
