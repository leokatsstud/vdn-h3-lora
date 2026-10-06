"""Snapshot the images currently present. Missing images are exclusions, never restored."""
import argparse,json,hashlib,os,struct,zipfile
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def select_records(manifest,dataset):
    dataset=Path(dataset).resolve();image_dir=dataset/'images'
    if not image_dir.is_dir():raise ValueError('Missing current images directory')
    rows={r['id']:r for r in manifest['records']}
    present={p.stem:p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}}
    unknown=sorted(set(present)-set(rows))
    if unknown:raise ValueError('Unmanifested images need explicit metadata: '+','.join(unknown))
    kept=[]
    for ident in sorted(present):
        r=rows[ident];image=dataset/r['video'];caption=dataset/r['caption']
        if not image.resolve().is_relative_to(dataset) or not caption.resolve().is_relative_to(dataset):raise ValueError('Path escapes dataset')
        if present[ident].resolve()!=image.resolve():raise ValueError('Unexpected image extension/path: '+ident)
        if digest(image)!=r['video_sha256']:raise ValueError('Retained image changed: '+ident)
        if not caption.exists() or digest(caption)!=r['caption_sha256']:raise ValueError('Retained caption missing/changed: '+ident)
        if r['media_kind']!='image' or r['frames']!=1 or r['references'] or r['audio_supervised']:raise ValueError('Not a plain image target: '+ident)
        kept.append(dict(r,caption_characters=len(caption.read_text(encoding='utf-8-sig').strip())))
    if not kept:raise ValueError('No remaining images')
    return kept,sorted(set(rows)-set(present))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--source-manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--previous-selection',type=Path);ap.add_argument('--archive',action='store_true');a=ap.parse_args()
    manifest=json.loads(a.source_manifest.read_text(encoding='utf-8-sig'));kept,excluded=select_records(manifest,a.dataset)
    a.out.mkdir(parents=True,exist_ok=True);target=a.out/'images_manifest.json'
    new=dict(manifest,records=kept,records_sha256=fingerprint(kept),dataset_root=os.path.relpath(a.dataset.resolve(),target.parent.resolve()).replace('\\','/'))
    target.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
    previous=set(json.loads(a.previous_selection.read_text(encoding='utf-8-sig'))['kept_ids']) if a.previous_selection else set(r['id'] for r in manifest['records'])
    now={r['id'] for r in kept}
    report={'images_directory':str((a.dataset/'images').resolve()),'kept':len(kept),'excluded_from_original':len(excluded),'newly_deleted_since_previous':sorted(previous-now),'reintroduced_since_previous':sorted(now-previous),'kept_ids':sorted(now),'excluded_ids':excluded,'remaining_source_videos':len({r['source_video_id'] for r in kept}),'records_sha256':new['records_sha256'],'remaining_media_and_caption_SHA_verified':True,'sampler':'all retained images exactly once per shuffled epoch; equal image weight'}
    if a.archive:
        archive=a.out/'current_selected_images.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
            for r in kept:
                for rel in [r['video'],r['caption']]:z.write(a.dataset/rel,rel)
        report['archive']={'name':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)}
    (a.out/'current_frame_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['kept','excluded_from_original','remaining_source_videos','records_sha256']}))
if __name__=='__main__':main()
