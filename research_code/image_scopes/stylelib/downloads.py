"""Bounded-memory, resumable scoped downloads; secret URLs are never logged."""
import base64,hashlib,os,stat,time,urllib.error,urllib.request,zipfile
from pathlib import Path,PurePosixPath

def safe_path(root,relative):
    rel=PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:raise ValueError('Unsafe relative path')
    root=Path(root).resolve();dest=(root/Path(*rel.parts)).resolve()
    if not dest.is_relative_to(root):raise ValueError('Download path escapes destination')
    return dest

def file_hash(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2*2**20),b''):h.update(chunk)
    return h.hexdigest()

def verify_file(path,item):
    if not path.exists() or path.stat().st_size!=item['bytes']:return False
    if item.get('sha256'):return file_hash(path)==item['sha256']
    if item.get('git_blob_sha1'):
        h=hashlib.sha1(f'blob {item["bytes"]}\0'.encode())
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(2*2**20),b''):h.update(chunk)
        return h.hexdigest()==item['git_blob_sha1']
    raise ValueError('Every download needs a content digest')

def download_one(root,item,stop_requested=lambda:False):
    path=safe_path(root,item['path']);path.parent.mkdir(parents=True,exist_ok=True)
    if verify_file(path,item):return 'verified_existing'
    if path.exists():raise ValueError('Existing completed download has wrong content: '+item['path'])
    if 'inline_base64' not in item and 'url' not in item:raise ValueError('Missing existing-only model file; regenerate a download plan')
    temp=path.with_name(path.name+'.part')
    if 'inline_base64' in item:
        temp.write_bytes(base64.b64decode(item['inline_base64'],validate=True))
    else:
        for attempt in range(5):
            if stop_requested():raise InterruptedError('Stop requested')
            offset=temp.stat().st_size if temp.exists() else 0
            if offset==item['bytes']:break
            if offset>item['bytes']:raise ValueError('Partial download too large')
            req=urllib.request.Request(item['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
            try:
                with urllib.request.urlopen(req,timeout=60) as response:
                    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise ValueError('Server did not honor resume offset')
                    with temp.open('ab' if offset else 'wb') as f:
                        while True:
                            if stop_requested():raise InterruptedError('Stop requested')
                            block=response.read(2*2**20)
                            if not block:break
                            f.write(block)
                            if f.tell()>item['bytes']:raise ValueError('Download exceeded declared size')
                if temp.stat().st_size==item['bytes']:break
            except urllib.error.HTTPError as exc:
                if exc.code in (401,403):raise RuntimeError('Scoped link expired or unauthorized; refresh plan and resume') from None
                if attempt==4:raise RuntimeError('Download HTTP failure') from None
            except (urllib.error.URLError,TimeoutError,OSError):
                if attempt==4:raise RuntimeError('Download network failure') from None
            time.sleep(min(2**attempt,8))
    if not verify_file(temp,item):raise ValueError('Downloaded content hash differs: '+item['path'])
    temp.replace(path);return 'downloaded_verified'

def extract_zip_safely(archive,destination,max_uncompressed_bytes=12*2**30):
    with zipfile.ZipFile(archive) as z:
        if sum(x.file_size for x in z.infolist())>max_uncompressed_bytes:raise ValueError('Archive exceeds extraction size limit')
        for item in z.infolist():
            dest=safe_path(destination,item.filename.rstrip('/'))
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Archive symlink forbidden')
            if item.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.extract.part')
            with z.open(item) as source,temp.open('wb') as target:
                for block in iter(lambda:source.read(2*2**20),b''):target.write(block)
            temp.replace(dest)
