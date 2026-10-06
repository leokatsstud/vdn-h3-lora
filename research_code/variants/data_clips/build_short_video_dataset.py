"""Local scene-aware dataset preparation. No AI, rental, training, network, or deletion.
Input remains unchanged. Crops use lossless H264 and PCM; timestamps/geometry are native.
"""
import argparse,collections,concurrent.futures,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'dataset_for_gpu';OUT=ROOT/'dataset_video_clips_h3'
FFMPEG='ffmpeg';FFPROBE='ffprobe'
ALLOWED=[39,56,73,90,107,124];SEED=20261005
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def probe(p):return json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(p)]))
def hash_key(ident):return hashlib.sha256(f'{SEED}:{ident}'.encode()).hexdigest()
def load_source(p):
 m=json.loads((SOURCE/'_metadata'/f'{p.stem}.json').read_text(encoding='utf-8'));info=probe(p);v=next(s for s in info['streams'] if s['codec_type']=='video');aud=next((s for s in info['streams'] if s['codec_type']=='audio'),None)
 n=int(v['nb_frames']);inv=m['original_inventory'];assert n==inv['frames'] and v['r_frame_rate']=='24/1'
 assert abs(float(v.get('start_time','0')))<.001,'Unexpected video start PTS'
 digest=sha(p);expected=json.loads((ROOT/'dataset_work/video_filter_retained_frames_20261005/completed_report.json').read_text(encoding='utf-8'))['original_video_sha256'][p.stem];assert digest==expected
 cuts=m.get('cuts',[]);assert cuts==sorted(set(cuts)) and all(0<c<n for c in cuts)
 review={}
 candidates=[ROOT/'dataset_final'/f'{p.stem}.cuts.review.json',ROOT/'dataset_work/portrait9_20260930/review'/p.stem/f'{p.stem}.cuts.review.json']
 for candidate in candidates:
  if candidate.exists():
   r=json.loads(candidate.read_text(encoding='utf-8'))
   if r.get('video_sha256')==digest:review={int(x['frame']):x for x in r.get('boundaries',[])};break
 transitions=m.get('transitions',{});bounds=[]
 for c in cuts:
  kind=transitions.get(str(c),'hard_cut');r=review.get(c,{});span=r.get('range_frames',[c,c] if kind=='hard_cut' else [c-3,c+3]);bounds.append({'frame':c,'kind':kind,'range':span,'range_verified':bool(r)})
 padding=int(inv.get('source_end_frames_padded',0));shots=[];raw=[0,*cuts,n-padding]
 for i,(a,b) in enumerate(zip(raw,raw[1:])):
  start=a if i==0 else max(a,bounds[i-1]['range'][1])+2
  end=b if i==len(raw)-2 else min(b,bounds[i]['range'][0])-2
  shots.append({'shot':i+1,'raw_start':a,'raw_end':b,'safe_start':start,'safe_end':max(start,end)})
 return {'id':p.stem,'file':str(p.resolve()),'sha256':digest,'frames':n,'width':int(v['width']),'height':int(v['height']),'video_stream':v,'audio_stream':aud,'padding':padding,'cuts':cuts,'boundaries':bounds,'shots':shots,'metadata':m}
def pick_balanced(rows,count):
 selected=[]
 for portrait in [False,True]:
  group=sorted([r for r in rows if (r['height']>r['width'])==portrait],key=lambda r:hash_key(r['id']))
  if group:selected.append(group[0]['id'])
 for r in sorted(rows,key=lambda r:hash_key(r['id'])):
  if len(selected)>=count:break
  if r['id'] not in selected:selected.append(r['id'])
 return set(selected)
def partition(length):
 # Maximise coverage; among equal coverage use fewer clips, then favour39.
 dp={0:[]}
 for total in range(1,length+1):
  candidates=[dp[total-n]+[n] for n in ALLOWED if total-n in dp]
  if candidates:dp[total]=min(candidates,key=lambda parts:(len(parts),-parts.count(39),tuple(parts)))
 covered=max(dp);return dp[covered],length-covered
def plan():
 if (OUT/'plan.json').exists():raise FileExistsError('Existing plan: inspect/resume it, do not silently replace')
 OUT.mkdir(exist_ok=True);files=sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in ['.mov','.mp4','.mkv','.webm']);rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(load_source,files),1):
   rows.append(row)
   if i%40==0:print('Inspected sources',i,flush=True)
 full=pick_balanced([r for r in rows if r['frames']>=260 and not r['padding']],12)|pick_balanced([r for r in rows if r['frames']==124 and not r['padding']],6)
 forced={r['id'] for r in rows if not any(s['safe_end']-s['safe_start']>=39 for s in r['shots'])};full|=forced
 clips=[];omitted=[]
 for r in rows:
  if r['id'] in full:
   clips.append({'id':f"{r['id']}_full_n{r['frames']:04d}",'source_id':r['id'],'start_frame':0,'end_frame_exclusive':r['frames'],'frames':r['frames'],'full_source':True,'preservation_reason':'all shots too short for39' if r['id'] in forced else 'deterministic full-video anchor','shot':None,'existing_artificial_tail_frames':r['padding']});continue
  for shot in r['shots']:
   start,end=shot['safe_start'],shot['safe_end'];length=end-start
   if length<39:
    omitted.append({'source_id':r['id'],'shot':shot['shot'],'start_frame':shot['raw_start'],'end_frame_exclusive':shot['raw_end'],'reason':'safe scene shorter than39; never merged/padded'});continue
   sizes,remainder=partition(length);cursor=start+remainder//2
   for j,n in enumerate(sizes,1):
    clips.append({'id':f"{r['id']}_s{shot['shot']:02d}_c{j:02d}_f{cursor:06d}_n{n:04d}",'source_id':r['id'],'start_frame':cursor,'end_frame_exclusive':cursor+n,'frames':n,'full_source':False,'shot':shot['shot'],'existing_artificial_tail_frames':0,'shot_safe_start':start,'shot_safe_end':end});cursor+=n
   if remainder:omitted.append({'source_id':r['id'],'shot':shot['shot'],'frames':remainder,'reason':'codec length remainder, trimmed without padding'})
 sources={r['id']:r for r in rows}
 for c in clips:
  r=sources[c['source_id']];c.update(width=r['width'],height=r['height'],fps=24,video_file=f"videos/{c['id']}.mov",caption_file=f"captions/{c['id']}.txt",context_file=f"caption_context/{c['id']}.json",sampling_group='short39' if c['frames']==39 else 'medium56to124' if c['frames']<=124 else 'long')
 p={'schema':1,'seed':SEED,'source_folder':str(SOURCE.resolve()),'sources':sources,'clips':clips,'omitted':omitted,'allowed_short_frames':ALLOWED,'hard_cut_guard_frames':2,'video_encoding':'libx264 qp0 lossless, native YUV/geometry/FPS','audio_encoding':'PCM retaining source rate/channels/decoded precision','no_new_padding':True,'no_source_changes':True}
 write(OUT/'plan.json',p)
 groups=collections.Counter(c['sampling_group'] for c in clips);sampling={'intended_presentation_probabilities':{'short39':.5,'medium56to124':.4,'long':.1},'clip_counts':dict(groups),'per_clip_probability':{c['id']:{'short39':.5,'medium56to124':.4,'long':.1}[c['sampling_group']]/groups[c['sampling_group']] for c in clips},'note':'Presentation probabilities are a future sampler proposal, not file-count proportions; no training started'};write(OUT/'sampling_plan.json',sampling)
 summary={'source_videos':len(rows),'represented_sources':len({c['source_id'] for c in clips}),'clips':len(clips),'frame_histogram':dict(collections.Counter(c['frames'] for c in clips)),'full_anchors':len(full),'forced_full_sources':sorted(forced),'omitted_entries':len(omitted),'output_frames':sum(c['frames'] for c in clips),'source_frames':sum(r['frames'] for r in rows)};write(OUT/'planning_summary.json',summary);print(json.dumps(summary),flush=True)
def audio_codec(fmt):
 return {'s16':'pcm_s16le','s16p':'pcm_s16le','s32':'pcm_s32le','s32p':'pcm_s32le','flt':'pcm_f32le','fltp':'pcm_f32le','dbl':'pcm_f64le','dblp':'pcm_f64le'}[fmt]
def encode(c,source,dest):
 if c['full_source']:shutil.copyfile(source['file'],dest);return
 vf=f"trim=start_frame={c['start_frame']}:end_frame={c['end_frame_exclusive']},setpts=PTS-STARTPTS"
 cmd=[FFMPEG,'-v','error','-nostdin','-threads','2','-i',source['file'],'-map','0:v:0','-vf',vf,'-c:v','libx264','-preset','veryfast','-qp','0','-threads','2','-pix_fmt',source['video_stream']['pix_fmt'],'-fps_mode','passthrough']
 if source['audio_stream']:
  a,b=c['start_frame']/24,c['end_frame_exclusive']/24
  cmd+=['-map','0:a:0','-af',f'atrim=start={a:.12f}:end={b:.12f},asetpts=PTS-{a:.12f}/TB','-c:a',audio_codec(source['audio_stream']['sample_fmt'])]
 for key,opt in [('color_space','-colorspace'),('color_transfer','-color_trc'),('color_primaries','-color_primaries'),('color_range','-color_range')]:
  value=source['video_stream'].get(key)
  if value and value not in ['unknown','unspecified']:cmd +=[opt,value]
 cmd+=['-map_metadata','-1','-avoid_negative_ts','disabled','-movflags','+faststart','-y',str(dest)]
 result=subprocess.run(cmd,capture_output=True)
 if result.returncode:raise RuntimeError(result.stderr.decode('utf-8','replace')[-1500:])
def frame_hashes(path,vf=None,pix_fmt=None):
 cmd=[FFMPEG,'-v','error','-threads','1','-i',str(path),'-map','0:v:0']
 if vf:cmd+=['-vf',vf]
 if pix_fmt:cmd+=['-pix_fmt',pix_fmt]
 cmd+=['-fps_mode','passthrough','-f','framemd5','-']
 lines=subprocess.check_output(cmd).decode().splitlines();return [line.rsplit(',',1)[1].strip() for line in lines if line and not line.startswith('#')]
def audio_hash(path,stream,start=None,end=None):
 codec=audio_codec(stream['sample_fmt']);cmd=[FFMPEG,'-v','error','-threads','1','-i',str(path),'-map','0:a:0']
 if start is not None:cmd+=['-af',f'atrim=start={start:.12f}:end={end:.12f}']
 cmd+=['-c:a',codec,'-f',codec.removeprefix('pcm_'),'-']
 h=hashlib.sha256();size=0;process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 while block:=process.stdout.read(2**20):h.update(block);size+=len(block)
 error=process.stderr.read();assert process.wait()==0,error.decode('utf-8','replace')[-1000:]
 return h.hexdigest(),size
def verify(c,source,path,exact_pixels=True):
 info=probe(path);v=next(s for s in info['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==c['frames'] and [int(v['width']),int(v['height'])]==[c['width'],c['height']] and v['r_frame_rate']=='24/1'
 assert abs(float(v.get('start_time','0')))<.001
 aud=next((s for s in info['streams'] if s['codec_type']=='audio'),None)
 assert bool(aud)==bool(source['audio_stream'])
 if aud:
  assert aud['sample_rate']==source['audio_stream']['sample_rate'] and aud['channels']==source['audio_stream']['channels']
  if not c['full_source']:assert audio_hash(source['file'],source['audio_stream'],c['start_frame']/24,c['end_frame_exclusive']/24)==audio_hash(path,source['audio_stream']),'Decoded audio samples differ'
 if c['full_source']:assert sha(path)==source['sha256']
 elif exact_pixels:
  reference=frame_hashes(source['file'],f"trim=start_frame={c['start_frame']}:end_frame={c['end_frame_exclusive']},setpts=PTS-STARTPTS",source['video_stream']['pix_fmt']);actual=frame_hashes(path,pix_fmt=source['video_stream']['pix_fmt']);assert len(actual)==c['frames'] and actual==reference,'Lossless decoded frame hashes differ'
 return {'sha256':sha(path),'bytes':path.stat().st_size,'verification_version':2,'frames_verified':True,'native_geometry_fps_verified':True,'decoded_pixels_equal_source':True if c['full_source'] or exact_pixels else None,'decoded_audio_samples_equal_source':True,'audio_stream':{k:aud.get(k) for k in ['codec_name','sample_rate','channels','sample_fmt','start_time','duration']} if aud else None}
def context(c,source):
 m=source['metadata'];shot=m.get('corrected_annotation',{}).get('shots',{}).get(f"shot_{c['shot']}") if c['shot'] else None
 return {'clip_id':c['id'],'source_video_id':c['source_id'],'source_frame_range_inclusive':[c['start_frame'],c['end_frame_exclusive']-1],'source_seconds_range':[c['start_frame']/24,c['end_frame_exclusive']/24],'clip_frames':c['frames'],'clip_duration_seconds':c['frames']/24,'old_caption_file':f"../source_descriptions/{c['source_id']}.txt",'old_source_metadata_file':f"../source_descriptions/{c['source_id']}.json",'old_shot_number':c['shot'],'old_shot_description':shot,'caption_status':'reused_unchanged_full_video' if c['full_source'] else 'new_caption_required','instruction':'Watch/listen to THIS clip. Old caption and shot description are context only: omit actions, characters, cuts and sounds outside this crop. Never copy full-video timestamps into a short clip. Preserve native three-field H3 caption format and vstyleh3 once after [Shot 1].','known_cut_quality':'existing best-effort reviewed boundaries; not human ground truth'}
def build(pilot=False):
 p=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));clips=p['clips'];sources=p['sources']
 for folder in ['videos','captions','caption_context','source_descriptions']: (OUT/folder).mkdir(exist_ok=True)
 for ident,r in sources.items():
  shutil.copyfile(SOURCE/f'{ident}.txt',OUT/'source_descriptions'/f'{ident}.txt');write(OUT/'source_descriptions'/f'{ident}.json',r['metadata'])
 if pilot:
  crops=[c for c in clips if not c['full_source']];selected=[]
  for portrait in [False,True]:
   group=[c for c in crops if (c['height']>c['width'])==portrait]
   if group:selected.append(max(group,key=lambda c:(c['frames'],Path(sources[c['source_id']]['file']).stat().st_size/sources[c['source_id']]['frames'])))
  selected.append(max([c for c in crops if c['frames']==39],key=lambda c:Path(sources[c['source_id']]['file']).stat().st_size/sources[c['source_id']]['frames']))
  clips=list({c['id']:c for c in selected}.values())
 done={};progress_path=OUT/'build_progress.json'
 if progress_path.exists():done=json.loads(progress_path.read_text(encoding='utf-8')).get('verified',{})
 remaining=[]
 for c in clips:
  source=sources[c['source_id']];dest=OUT/c['video_file']
  if c['id'] in done and dest.exists() and sha(dest)==done[c['id']]['sha256']:
   if done[c['id']].get('verification_version')!=2:done[c['id']]=verify(c,source,dest)
   continue
  remaining.append(c)
 def process_clip(c):
  source=sources[c['source_id']];dest=OUT/c['video_file']
  assert dest.resolve().is_relative_to(OUT.resolve())
  if shutil.disk_usage(OUT).free<3*2**30:raise RuntimeError('Less than3GiB free: preserve completed clips, do not fill system disk')
  tmp=dest.with_name(dest.stem+'.partial.mov');encode(c,source,tmp);proof=verify(c,source,tmp);tmp.replace(dest)
  write(OUT/c['context_file'],context(c,source))
  if c['full_source']:shutil.copyfile(SOURCE/f"{c['source_id']}.txt",OUT/c['caption_file'])
  return c,proof
 with concurrent.futures.ThreadPoolExecutor(max_workers=1 if pilot else 3) as pool:
  for index,(c,proof) in enumerate(pool.map(process_clip,remaining),1):
   done[c['id']]=proof
   write(progress_path,{'stage':'pilot' if pilot else 'building','verified':done,'last_clip':c['id'],'total_planned':len(p['clips']),'verified_count':len(done),'CPU_workers':1 if pilot else 3})
   if index%10==0 or pilot:print(f"Verified clips {len(done)}/{len(p['clips'])}: {c['id']}",flush=True)
 if pilot:
  cropped=[(next(c for c in p['clips'] if c['id']==ident),proof) for ident,proof in done.items() if not next(c for c in p['clips'] if c['id']==ident)['full_source']]
  ratio=max(proof['bytes']/c['frames'] for c,proof in cropped)
  estimate=sum(Path(sources[c['source_id']]['file']).stat().st_size if c['full_source'] else math.ceil(c['frames']*ratio) for c in p['clips']);write(OUT/'pilot_report.json',{'verified_clips':len(clips),'conservative_estimated_bytes':estimate,'free_bytes':shutil.disk_usage(OUT).free,'largest_sample_bytes_per_frame':ratio,'lossless_decoded_pixels_verified':True});print(json.dumps({'pilot':len(clips),'estimate_GiB':round(estimate/2**30,2),'free_GiB':round(shutil.disk_usage(OUT).free/2**30,2)}),flush=True);return
 assert len(done)==len(p['clips'])
 records=[]
 for c in p['clips']:
  assert (OUT/c['video_file']).is_file() and (OUT/c['context_file']).is_file();records.append(dict(c,**done[c['id']],caption_status='ready_reused_full' if c['full_source'] else 'pending_new_caption'))
 (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
 write(OUT/'completion_report.json',{'completed':True,'clips':len(records),'represented_sources':len(sources),'total_bytes':sum(r['bytes'] for r in records),'full_caption_reused':sum(c['full_source'] for c in records),'new_captions_required':sum(not c['full_source'] for c in records),'all_frame_counts_geometry_FPS_verified':True,'all_decoded_crop_frames_equal_source':True,'all_decoded_audio_samples_equal_source':True,'all_full_videos_SHA_equal_source':True,'training_ready':False,'no_source_files_changed':True})
 write(progress_path,{'stage':'completed','verified':done,'total_planned':len(p['clips']),'verified_count':len(done)});print('Dataset complete; new crop captions still required.',flush=True)
def main():
 global ROOT,SOURCE,OUT,FFMPEG,FFPROBE
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','pilot','build']);ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ffmpeg',default='ffmpeg');ap.add_argument('--ffprobe',default='ffprobe');a=ap.parse_args()
 ROOT=a.workspace.resolve();SOURCE=ROOT/'dataset_for_gpu';OUT=a.out.resolve();FFMPEG=a.ffmpeg;FFPROBE=a.ffprobe
 if a.mode=='plan':plan()
 else:build(a.mode=='pilot')
if __name__=='__main__':main()
