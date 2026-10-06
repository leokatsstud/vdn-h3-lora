"""Create a fresh index over already encoded retained images; no model loads/recaching."""
import argparse,json,copy
from pathlib import Path
from stylelib.common import read_json,write_json,fingerprint,load_manifest,verify_record
def subset_index(source,manifest,source_cache):
    if not source['complete']:raise ValueError('Source cache incomplete')
    by_id={e['record']['id']:e for e in source['records']}
    if len(by_id)!=len(source['records']):raise ValueError('Duplicate source cache IDs')
    kept=[]
    for r in manifest['records']:
        e=copy.deepcopy(by_id[r['id']])
        if e['record']!=r:raise ValueError('Cached record differs: '+r['id'])
        for key in ['media']:
            path=(Path(source_cache)/e[key]).resolve()
            if not path.exists():raise ValueError('Missing cached tensor: '+str(path))
            e[key]=str(path)
        for mode,value in list(e['conditions'].items()):
            path=(Path(source_cache)/value).resolve()
            if not path.exists():raise ValueError('Missing cached text: '+str(path))
            e['conditions'][mode]=str(path)
        kept.append(e)
    return dict(source,records=kept,records_sha256=manifest['records_sha256'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-cache',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.resolve()==a.source_cache.resolve():raise ValueError('Do not overwrite the original cache')
    manifest,root=load_manifest(a.manifest)
    for r in manifest['records']:verify_record(r,root)
    index=subset_index(read_json(a.source_cache/'index.json'),manifest,a.source_cache)
    write_json(a.out/'index.json',index)
    write_json(a.out/'selection_verified.json',{'selected_images':len(index['records']),'records_sha256':manifest['records_sha256'],'source_cache':str(a.source_cache.resolve()),'recached':False})
    print('Fresh cache index prepared for retained images only; original cache untouched.')
if __name__=='__main__':main()
