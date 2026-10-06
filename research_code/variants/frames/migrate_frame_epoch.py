"""CPU-only explicit sampling-policy transition after the writer has saved/exited."""
import argparse,copy,shutil
from pathlib import Path
import torch
from stylelib import GPU_ROOT
from stylelib.common import read_json,write_json,sha256,fingerprint,atomic_torch_save,code_fingerprint
from stylelib.frame_epoch_sampler import FrameEpochSampler

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    path=a.train/'last.pt';source_sha=sha256(path);saved=torch.load(path,map_location='cpu',weights_only=True)
    previous=copy.deepcopy(saved['resume_identity_inputs'])
    if fingerprint(previous)!=saved['identity'] or saved['balanced_sampler']['version']!=1:raise ValueError('Not the reviewed source-balanced checkpoint')
    if len(saved['optimizer']['state'])!=600:raise ValueError('Incomplete optimizer')
    preserved=a.train/f'resume_before_epoch_change_{saved["step"]:04d}.pt'
    if preserved.exists():raise ValueError('Migration already exists')
    shutil.copy2(path,preserved)
    pool=[e for e in read_json(a.cache/'index.json')['records'] if e['record']['split']=='train']
    sampler=FrameEpochSampler(pool,0)
    sampler.rng.setstate(saved['balanced_sampler']['rng']);sampler.order=list(range(len(pool)));sampler.rng.shuffle(sampler.order)
    saved['balanced_sampler']=sampler.state_dict()
    current=copy.deepcopy(previous);current['sampler']='all_frames_shuffled_epoch_v2'
    files=['train_style_vdn.py','stylelib/model.py','stylelib/adapter.py','stylelib/packing.py','stylelib/objective.py','stylelib/sampling.py','stylelib/resume_policy.py','stylelib/media.py','stylelib/frame_epoch_sampler.py','stylelib/ledger.py','stylelib/stage_transition.py','stylelib/cache_policy.py','stylelib/distributed.py']
    current['trainer_code']=code_fingerprint(files)
    old_identity=saved['identity'];saved['identity']=fingerprint(current);saved['resume_identity_inputs']=current
    atomic_torch_save(saved,path)
    report={'step':saved['step'],'source_checkpoint_sha256':source_sha,'old_identity':old_identity,'new_identity':saved['identity'],
        'allowed_changes':['sampler policy','trainer sampler import/epoch log/checkpoint cadence'],
        'frames_per_epoch':len(pool),'repeat_within_epoch':False,'new_epoch_starts_at_this_checkpoint':True,
        'checkpoint_every':500,'immutable_500_milestones':True,'masters_optimizer_and_model_rng_preserved':True,
        'sampler_rng_preserved_before_new_permutation':True,'checkpoint_sha256':sha256(path)}
    write_json(a.train/'frame_epoch_migration.json',report)
    print(report)

if __name__=='__main__':main()
