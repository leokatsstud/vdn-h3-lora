"""Build a training manifest from image and video records."""
import argparse,json,os,subprocess
from pathlib import Path
from stylelib.common import read_json,write_json,sha256,verify_record,fingerprint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--split-map',type=Path,help='Optional explicit mapping id -> {source_group, split}; no random per-clip split')
    args=ap.parse_args();root=args.dataset.resolve();selection=read_json(root/'_metadata/curated_manifest.json');rows=[]
    registered={r['video'] for r in selection['records']}
    unknown=[p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.mov','.mp4','.mkv','.webm') and p.name not in registered]
    if unknown:raise ValueError('New videos are not registered in curated metadata; prepare captions/frame-grid metadata before caching: '+', '.join(sorted(unknown)))
    splits=read_json(args.split_map) if args.split_map else None
    for item in selection['records']:
        if not (root/item['video']).exists():continue
        item=dict(item)
        if item.get('reference_caption'):
            refmeta=read_json((root/item['reference_caption']).with_suffix('.json'))
            item['reference_caption_sha256']=refmeta['caption_sha256']
        verify_record(item,root);meta=read_json(root/item['metadata']);inv=meta['original_inventory']
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames','-of','json',str(root/item['video'])],text=True))['streams'][0]
        for k in ('width','height'):
            if int(info[k])!=inv[k]:raise ValueError(item['id']+': geometry changed')
        if info.get('nb_frames') in (None,'N/A'):raise ValueError('Container frame count unavailable; decode-audit this file before cache creation: '+item['id'])
        if int(info['nb_frames'])!=inv['frames'] or inv['frames']%17!=5:raise ValueError('Invalid frame grid: '+item['id'])
        for key in ('avg_frame_rate','r_frame_rate'):
            n,d=map(int,info[key].split('/'))
            if n/d!=24:raise ValueError('Not 24 fps: '+item['id'])
        row={k:item[k] for k in ['id','video','caption','video_sha256','caption_sha256','references','reference_caption']}
        row.update(width=inv['width'],height=inv['height'],frames=inv['frames'],fps=24,
                   source_id=inv['source_id'],source_group='unverified_overlap_pool',split='train',
                   artificial_tail_frames=inv.get('source_end_frames_padded',0),audio_present=inv['audio_present'],
                   audio_supervised=bool(item.get('audio_supervised',False)),audio_approval=item.get('audio_approval','pending'))
        if row['audio_supervised'] and row['audio_approval']!='approved_clean_synced':raise ValueError('Audio data loss requires reviewed clean-synced approval')
        if row['reference_caption']:row['reference_caption_sha256']=sha256(root/row['reference_caption'])
        if splits:
            if row['id'] not in splits:raise ValueError('Split mapping is incomplete')
            row.update({k:splits[row['id']][k] for k in ('source_group','split')})
        rows.append(row)
    if not rows:raise ValueError('Empty selection')
    groups={}
    for row in rows:
        if row['split'] not in ('train','validation'):raise ValueError('Invalid split')
        if row['source_group'] in groups and groups[row['source_group']]!=row['split']:raise ValueError('A source group crosses train/validation')
        groups[row['source_group']]=row['split']
    out=args.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    result={'schema':2,'dataset_root':os.path.relpath(root,out.parent).replace('\\','/'),
            'split_policy':'explicit_group_map' if splits else 'all_train_external_generation_validation',
            'independent_video_holdout':bool(splits),'records':rows}
    result['records_sha256']=fingerprint(rows);write_json(out,result)
    print(json.dumps({'manifest':str(out),'videos':len(rows),'train':sum(r['split']=='train' for r in rows),'references':sum(len(r['references']) for r in rows),'split_policy':result['split_policy']}))
if __name__=='__main__':main()
