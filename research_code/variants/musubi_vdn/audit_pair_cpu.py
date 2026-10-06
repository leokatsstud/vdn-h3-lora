"""Read-only CPU comparison of initial adapters and actual first500 presentations."""
import argparse,json,torch
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--vdn',type=Path,required=True);ap.add_argument('--dense',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();vdn=args.vdn;dense=args.dense
a=torch.load(vdn/'resume_0000.pt',map_location='cpu',weights_only=True)['style'];b=torch.load(dense/'resume_0000.pt',map_location='cpu',weights_only=True)['style']
a={k.replace('.attn.orig.','.attn.'):v for k,v in a.items()}
assert set(a)==set(b)
initial_equal=all(torch.equal(a[k],b[k]) for k in a)
equal_A=sum(torch.equal(a[k],b[k]) for k in a if k.endswith('.style_A'))
def events(p):return [json.loads(line) for line in (p/'events.jsonl').read_text().splitlines()][:500]
x,y=events(vdn),events(dense);assert len(x)==len(y)==500
report={'first500_same_image_order':all(i['id']==j['id'] for i,j in zip(x,y)),'first500_same_noise_levels':all(i['u']==j['u'] for i,j in zip(x,y)),'first500_same_LR':all(i['lr']==j['lr'] for i,j in zip(x,y)),'initial_FP32_LoRA_tensors_equal':initial_equal,'equal_initial_A_matrices':equal_A,'initial_A_matrices':sum(k.endswith('.style_A') for k in a),'all_initial_B_zero':all(torch.count_nonzero(s[k])==0 for s in [a,b] for k in s if k.endswith('.style_B')),'scope':'CPU read of immutable step0 states and events; no GPU calls or training'}
args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
