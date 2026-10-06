"""Preserve sampler, adapter and optimizer state when excluding deleted frames."""
import argparse,copy,json,os,shutil
from pathlib import Path
from stylelib import GPU_ROOT
from stylelib.common import read_json,write_json,fingerprint,sha256,atomic_torch_save,verify_record
from stylelib.balanced_sampler import SourceBalancedSampler

def filter_sampler(old,pool,new_pool):
    mapping={e['record']['id']:i for i,e in enumerate(new_pool)}
    remap={i:mapping[e['record']['id']] for i,e in enumerate(pool) if e['record']['id'] in mapping}
    sampler=SourceBalancedSampler(new_pool,0);state=copy.deepcopy(old)
    if set(state['frame_orders'])!=set(sampler.groups):raise ValueError('Subset must retain every source')
    state['identity']=sampler.identity
    for key,order in old['frame_orders'].items():
        state['frame_orders'][key]=[remap[i] for i in order if i in remap]
        state['frame_cursors'][key]=sum(i in remap for i in order[:old['frame_cursors'][key]])
    sampler.load_state_dict(state)
    return sampler.state_dict()

def migrate(root,old,selection,expected_sha):
    import torch
    from safetensors import safe_open
    work=root/'work';work.mkdir(parents=True,exist_ok=True)
    original=read_json(old/'work/images_manifest.json');index=read_json(old/'work/cache_frames/index.json')
    selected=set(selection['kept_ids']);records=[r for r in original['records'] if r['id'] in selected]
    if len(records)!=len(selected) or len({r['source_video_id'] for r in records})!=175:raise ValueError('Selection/source mismatch')
    dataset=(old/'work'/original['dataset_root']).resolve()
    for r in records:verify_record(r,dataset)
    manifest=dict(original,records=records,records_sha256=fingerprint(records),dataset_root=str(dataset))
    cache=work/'cache_frames';cache.mkdir(exist_ok=True)
    entries=[e for e in index['records'] if e['record']['id'] in selected]
    for e in entries:
        for name in [e['media'],e.get('visual'),*e['conditions'].values()]:
            if name and not (cache/name).exists():os.link(old/'work/cache_frames'/name,cache/name)
    fresh=dict(index,records=entries,records_sha256=manifest['records_sha256'])
    write_json(work/'images_manifest.json',manifest);write_json(cache/'index.json',fresh)
    source=old/'work/results/images_step100/resume_images_0100.pt'
    if sha256(source)!=expected_sha:raise ValueError('Source checkpoint SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    if saved['step']!=100 or len(saved['optimizer']['state'])!=600:raise ValueError('Invalid step100 optimizer')
    # Prove that the downloaded final adapter is the BF16 projection of this checkpoint.
    with safe_open(str(old/'work/results/images_step100/style_images_0100.safetensors'),framework='pt') as f:
        for name,t in saved['style'].items():
            key=name.replace('.style_A','.lora_A.weight').replace('.style_B','.lora_B.weight')
            if not torch.equal(t.to(torch.bfloat16),f.get_tensor(key)):raise ValueError('Final adapter tensor differs from committed master')
    previous=copy.deepcopy(saved['resume_identity_inputs'])
    if fingerprint(previous)!=saved['identity']:raise ValueError('Checkpoint identity invalid')
    current=copy.deepcopy(previous);current['records']=manifest['records_sha256']
    if current['cache_settings']!=fresh['settings']:raise ValueError('Encoding recipe changed')
    pool=[e for e in index['records'] if e['record']['split']=='train']
    new_pool=[e for e in entries if e['record']['split']=='train']
    saved['balanced_sampler']=filter_sampler(saved['balanced_sampler'],pool,new_pool)
    old_identity=saved['identity'];saved['resume_identity_inputs']=current;saved['identity']=fingerprint(current)
    train=work/'images/train';train.mkdir(parents=True,exist_ok=True)
    if (train/'last.pt').exists():raise ValueError('Do not remigrate over continuation checkpoint')
    atomic_torch_save(saved,train/'last.pt')
    shutil.copy2(old/'work/images/train/events.jsonl',train/'events.jsonl')
    report={'source_checkpoint_sha256':expected_sha,'old_identity':old_identity,'new_identity':saved['identity'],
        'source_step':100,'remaining_frames':len(records),'excluded_ids':selection['excluded_ids'],
        'optimizer_master_and_noise_rng_preserved':True,'sampler_policy':'random source cycles and random frame cycles; deleted indices removed without RNG reset',
        'final_adapter_tensor_equivalence_verified':True,'prior_generation_file_sha_issue_not_retroactively_resolved':True,
        'allowed_identity_change':['records'],'new_checkpoint_sha256':sha256(train/'last.pt')}
    write_json(work/'cleaning_migration.json',report)
    write_json(train/'cleaning_migration.json',report)
    del saved
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    args=parser.parse_args()
    migrate(args.root,args.old,read_json(args.selection),args.checkpoint_sha256)

if __name__=='__main__':main()
