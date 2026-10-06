import hashlib,json,os
from pathlib import Path
from . import GPU_ROOT

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def code_fingerprint(files):return fingerprint({f:sha256(GPU_ROOT/f) for f in files})
def contract():return read_json(GPU_ROOT/'pilot_contract.json')
def load_manifest(path):
    path=Path(path).resolve();data=read_json(path)
    if fingerprint(data['records'])!=data['records_sha256']:raise ValueError('Manifest content changed without rebuilding its identity')
    root=(path.parent/data['dataset_root']).resolve()
    return data,root
def atomic_torch_save(value,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)
def cache_key(record,settings):
    return fingerprint({'record':record,'settings':settings,'cache_schema':2})
def verify_record(record,root):
    pairs=[(record['video'],record['video_sha256']),(record['caption'],record['caption_sha256'])]
    pairs += [(r['path'],r['sha256']) for r in record['references']]
    if record.get('reference_caption'):pairs.append((record['reference_caption'],record['reference_caption_sha256']))
    for rel,expected in pairs:
        path=(Path(root)/rel).resolve()
        if not path.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes dataset')
        if sha256(path)!=expected:raise ValueError(f'Input changed after manifest creation: {rel}')
