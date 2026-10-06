"""One H200/B200, actual videos, frozen VDN-DMD, 85/15 T2V/reference pilot."""
import argparse,contextlib,json,random,signal,time
from pathlib import Path
import torch
from stylelib import GPU_ROOT
from stylelib.common import read_json,write_json,load_manifest,contract,fingerprint,atomic_torch_save,verify_record,code_fingerprint
from stylelib.adapter import style_parameters,style_state,load_style_state,export_style,style_strength,style_modules
from stylelib.objective import paired_times,guidance_target,style_loss
from stylelib.packing import PackedCondition,noise_sample,predict
from stylelib.model import build_style_model,set_inner_checkpoints
from stylelib.sampling import effective_reference_fraction
from stylelib.resume_policy import authorize_resume
from stylelib.media import validate_phase_records
from stylelib.frame_epoch_sampler import FrameEpochSampler
from stylelib.ledger import reconcile_events
from stylelib.stage_transition import validate_stage_transition
from stylelib.cache_policy import select_cache_contract
from stylelib.distributed import setup,any_rank,average_adapter_gradients,gather,barrier
from stylelib.video_av_policy import presentation_mode,put_first_reference_first
from stylelib.experiment_policy import validate_fresh_seconds,fresh_window_finished,numbered_checkpoint_due

def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
    return obj
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile',choices=['h200','b200','h100_dual'],required=True)
    ap.add_argument('--phase',choices=['images','videos'],required=True)
    ap.add_argument('--init-from',type=Path,help='Explicit image-to-video initialization; not same-run resume')
    ap.add_argument('--image-review',type=Path,help='Reviewed image checkpoint authorization for video stage')
    ap.add_argument('--steps',type=int,default=100);ap.add_argument('--hours',type=float,default=10);ap.add_argument('--seed',type=int,default=20260929)
    ap.add_argument('--resume',type=Path);ap.add_argument('--device',default='cuda:0');ap.add_argument('--no-activation-offload',action='store_true')
    ap.add_argument('--purpose',choices=['smoke','pilot'],default='pilot')
    ap.add_argument('--reference-fraction',type=float,default=0,help='This staged style experiment is text-only; must remain0')
    ap.add_argument('--checkpoint-every',type=int,default=5,help='Refresh portable last.pt and style_last.safetensors every N updates; milestone files remain separate')
    ap.add_argument('--reviewed-through-step',type=int,default=0,help='Explicit quality-gate authorization for continuation beyond step100')
    ap.add_argument('--allow-audio-supervision',action='store_true',help='Enable target-audio loss during video training')
    ap.add_argument('--video-training-budget-seconds',type=float,default=None,help='Actual video update seconds across resumes; exclude loading/generations')
    ap.add_argument('--fresh-experiment-seconds',type=float,help='Explicitly authorized image experiment; measured after model loading, at most3600seconds')
    ap.add_argument('--learning-rate',type=float,default=None,help='Explicit optimizer override; does not change media/text cache math')
    ap.add_argument('--allow-learning-rate-change',action='store_true',help='Permit a narrowly checked LR-only resume migration; preserve optimizer and RNG')
    ap.add_argument('--prior-runs',type=Path,nargs='*',default=[],help='Completed smoke/earlier run folders charged to the same 10-hour budget')
    args=ap.parse_args()
    if args.steps<1 or not 0<args.hours<=10:ap.error('Positive steps and at most 10 hours required')
    if args.steps>10000:ap.error('At most10000updates per staged experiment; pipeline windows remain bounded')
    validate_fresh_seconds(args.fresh_experiment_seconds,args.phase,args.purpose,args.init_from)
    if args.steps>100 and args.fresh_experiment_seconds is None and (args.reviewed_through_step<(10 if args.phase=='videos' else 100) or args.resume is None):ap.error('Beyond step100 requires a reviewed milestone>=100 and explicit resume')
    if args.checkpoint_every<1:ap.error('checkpoint-every must be positive')
    if args.init_from and (args.phase!='videos' or args.resume):ap.error('init-from is only for a NEW video phase')
    if args.phase=='videos' and args.resume is None and (args.init_from is None or args.image_review is None):ap.error('Video phase requires an image checkpoint and reviewed authorization')
    if args.phase=='images' and (args.reference_fraction!=0 or args.allow_audio_supervision):ap.error('Image phase remains text-only without audio supervision')
    if args.video_training_budget_seconds is not None and (args.phase!='videos' or not 0<args.video_training_budget_seconds<=14400):ap.error('Video update budget must be within(0,14400]')
    if args.learning_rate is not None and not 0<args.learning_rate<=1e-4:ap.error('Experimental LR override must be within (0,1e-4]')
    from filelock import FileLock
    args.out.mkdir(parents=True,exist_ok=True)
    # Singletons are enforced by the pipeline's global lock. Each replica gets
    # its own worker lock; two unrelated launches cannot share the output.
    with FileLock(str(args.out/f'.worker_rank{int(__import__("os").environ.get("RANK","0"))}.lock'),timeout=0):run(args)

def run(args):
    rank,world,args.device=setup(args.device)
    if world>1 and args.phase!='images':raise ValueError('Dual H100 replicas are frame-stage only; full-video memory sharding is not implemented')
    if (args.profile=='h100_dual')!=(world==2):raise ValueError('h100_dual must be launched with exactly2torchrun workers')
    started=time.perf_counter();cfg=contract();manifest,root=load_manifest(args.manifest);index=read_json(args.cache/'index.json')
    prior_seconds=0.
    for path in {p.resolve() for p in args.prior_runs}:
        if path==args.out.resolve():raise ValueError('Current run cannot also be a prior run')
        prior=read_json(path/'progress.json')
        if prior['status']=='running':raise ValueError('A prior run is still active')
        prior_seconds+=prior['elapsed_training_seconds']
    if not index['complete'] or index['records_sha256']!=manifest['records_sha256']:raise ValueError('Cache incomplete or manifest changed')
    cache_cfg=select_cache_contract(cfg,index['settings'],args.phase,read_json(GPU_ROOT/'video_cache_contract.json'))
    if index['settings']['sources']!=read_json(args.weights/'sources.json'):raise ValueError('Cache/model weight pins differ')
    for row in manifest['records']:verify_record(row,root)
    pool=[e for e in index['records'] if e['record']['split']=='train'];refpool=[e for e in pool if 'ref' in e['conditions']]
    validate_phase_records([e['record'] for e in pool],args.phase,args.allow_audio_supervision)
    if not pool:raise ValueError('Empty training pool')
    ref_fraction=effective_reference_fraction(args.reference_fraction,cfg['sampling']['reference_presentation_fraction'],len(refpool))
    profile=read_json(GPU_ROOT/'profiles'/f'{args.profile}.json')
    offload=profile['activation_offload'] and not args.no_activation_offload
    effective_lr=cfg['optimizer']['learning_rate'] if args.learning_rate is None else args.learning_rate
    identity_inputs={'records':manifest['records_sha256'],'cache_settings':index['settings'],'contract':cfg,'profile':profile,'offload':offload,'seed':args.seed,'purpose':args.purpose,'reference_fraction':ref_fraction,'effective_learning_rate':effective_lr,'training_phase':args.phase,'world_size':world,'sampler':'all_frames_shuffled_epoch_v2','fresh_experiment_seconds':args.fresh_experiment_seconds,'audio_supervision_authorized':args.allow_audio_supervision,'video_training_budget_seconds':args.video_training_budget_seconds,'reference_policy':'eligible_probability_with_first_ref_video','trainer_code':code_fingerprint(['train_style_vdn.py','stylelib/model.py','stylelib/adapter.py','stylelib/packing.py','stylelib/objective.py','stylelib/sampling.py','stylelib/resume_policy.py','stylelib/media.py','stylelib/frame_epoch_sampler.py','stylelib/ledger.py','stylelib/stage_transition.py','stylelib/cache_policy.py','stylelib/distributed.py','stylelib/experiment_policy.py','stylelib/video_av_policy.py'])}
    identity=fingerprint(identity_inputs)
    progress_path=args.out/'progress.json'
    if (args.out/'last.pt').exists() and args.resume is None:raise FileExistsError('Existing run: pass --resume last.pt or choose another output folder')
    if (args.out/'STOP').exists():raise RuntimeError('STOP exists; remove it only to intentionally resume')
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    torch.cuda.set_device(device);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    # Sequential CPU initialization prevents two replicas doubling temporary
    # host-RAM requirements while both still retain their own GPU model.
    for owner in range(world):
        if owner==rank:model,model_info=build_style_model(args.weights,cfg,str(device),True,profile['softmax_backend'])
        barrier(world)
    if world>1:
        from stylelib.common import sha256
        import hashlib
        initial=hashlib.sha256()
        for p in style_parameters(model):initial.update(p.detach().cpu().numpy().tobytes())
        if len(set(gather(initial.hexdigest(),world)))!=1:raise ValueError('Replica style initialization differs')
    params=style_parameters(model);optcfg=dict(cfg['optimizer'],learning_rate=effective_lr)
    optimizer=torch.optim.AdamW(params,lr=optcfg['learning_rate'],betas=tuple(optcfg['betas']),eps=optcfg['eps'],weight_decay=optcfg['weight_decay'],foreach=False)
    rng=random.Random(args.seed+rank*100003);noise_rng=torch.Generator(device=device).manual_seed(args.seed+1+rank*100003)
    sampler=FrameEpochSampler(pool,args.seed+2)
    if args.phase=='videos' and ref_fraction>0 and args.resume is None:put_first_reference_first(sampler,pool)
    step=0;elapsed_before=0.;global_offset=0;transition=None;stopping={'requested':False}
    training_window_start=None;fresh_elapsed_before=0.;video_update_seconds=0.
    def stop(signum,frame):stopping['requested']=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.init_from:
        from stylelib.common import sha256
        review=read_json(args.image_review)
        if not review.get('approved_for_video') or review.get('checkpoint_sha256')!=sha256(args.init_from):raise ValueError('Video authorization does not identify the source checkpoint')
        saved=torch.load(args.init_from,map_location='cpu',weights_only=True)
        if review.get('committed_step')!=saved['step']:raise ValueError('Video authorization step differs')
        transition=validate_stage_transition(saved,cfg,model_info)
        if saved['effective_optimizer']!=optcfg:raise ValueError('Phase transition must retain the effective optimizer/LR')
        load_style_state(model,saved['style']);optimizer.load_state_dict(saved['optimizer'])
        global_offset=saved.get('global_update_count',saved['step'])
        if rank==0:write_json(args.out/'image_to_video_transition.json',dict(transition,source_checkpoint_sha256=review['checkpoint_sha256'],destination_seed=args.seed))
        del saved
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved.get('fresh_experiment_seconds')!=args.fresh_experiment_seconds:raise ValueError('Fresh-hour window mode differs on resume')
        fresh_elapsed_before=float(saved.get('fresh_window_elapsed_seconds',0.))
        video_update_seconds=float(saved.get('video_update_seconds',0.))
        if args.steps>100 and saved['step']<args.reviewed_through_step:raise ValueError('Reviewed milestone exceeds committed checkpoint step')
        migration=authorize_resume(saved,identity_inputs,args.allow_learning_rate_change)
        if migration:
            if (args.out/'run.json').exists():write_json(args.out/'run_before_lr_migration.json',read_json(args.out/'run.json'))
            write_json(args.out/'lr_migration.json',migration)
        if saved.get('prior_training_seconds',0.)!=prior_seconds:raise ValueError('Resume must retain the same prior-run budget accounting')
        load_style_state(model,saved['style']);optimizer.load_state_dict(saved['optimizer'])
        states=saved['replica_rng_states']
        if len(states)!=world:raise ValueError('Resume replica RNG count differs')
        local_state=states[rank]
        rng.setstate(local_state['sample_rng']);noise_rng.set_state(local_state['noise_rng']);torch.set_rng_state(local_state['torch_rng'])
        torch.cuda.set_rng_state(local_state['cuda_rng'],device=device);step=saved['step'];elapsed_before=saved['elapsed_training_seconds']
        sampler.load_state_dict(saved['balanced_sampler']);global_offset=saved['global_update_count']-step
        transition=saved.get('phase_transition');del saved
    if rank==0:reconcile_events(args.out/'events.jsonl',step)
    barrier(world)
    def elapsed():return elapsed_before+time.perf_counter()-started
    def fresh_elapsed():return fresh_elapsed_before+(0. if training_window_start is None else time.perf_counter()-training_window_start)
    def save(label):
        states=gather({'sample_rng':rng.getstate(),'noise_rng':noise_rng.get_state().cpu(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device).cpu()},world)
        if rank!=0:
            barrier(world);return
        if label!='last' and ((args.out/f'resume_{label}.pt').exists() or (args.out/f'style_{label}.safetensors').exists()):raise FileExistsError('Immutable checkpoint already exists: '+label)
        state={'identity':identity,'resume_identity_inputs':identity_inputs,'effective_optimizer':optcfg,'step':step,'style':style_state(model),'optimizer':cpu_tree(optimizer.state_dict()),'sample_rng':rng.getstate(),
               'noise_rng':noise_rng.get_state().cpu(),'torch_rng':torch.get_rng_state(),'cuda_rng':[states[0]['cuda_rng']],
               'elapsed_training_seconds':elapsed(),'prior_training_seconds':prior_seconds,'model_info':model_info,'config':cfg,'purpose':args.purpose,'training_phase':args.phase,'global_update_count':global_offset+step,'balanced_sampler':sampler.state_dict(),'phase_transition':transition,'replica_rng_states':states,'world_size':world}
        state.update(fresh_experiment_seconds=args.fresh_experiment_seconds,fresh_window_elapsed_seconds=fresh_elapsed(),video_update_seconds=video_update_seconds)
        atomic_torch_save(state,args.out/'last.pt')
        export_style(model,args.out/f'style_{label}.safetensors',{'rank':cfg['style']['rank'],'alpha':cfg['style']['alpha'],'step':step,'identity':identity,'vdn_revision':cfg['hf_revision'],'targets':'hybrid attn.orig QKVO + ff.net.0.proj/ff.net.2','experimental_objective':cfg['experiment']})
        if label!='last':
            if args.fresh_experiment_seconds is not None:
                import os
                os.link(args.out/'last.pt',args.out/f'resume_{label}.pt')
            else:atomic_torch_save(state,args.out/f'resume_{label}.pt')
        barrier(world)
    if rank==0:write_json(args.out/'run.json',{'identity':identity,'resume_identity_inputs':identity_inputs,'model':model_info,'config':cfg,'effective_optimizer':optcfg,'profile':profile,'offload':offload,'purpose':args.purpose,'training_phase':args.phase,'global_update_offset':global_offset,'reference_fraction':ref_fraction,'audio_supervised_records':sum(e['record']['audio_supervised'] for e in pool),'gpu':torch.cuda.get_device_name(device),'world_size':world,'split_policy':manifest['split_policy'],'cache_gpu_seconds':index['gpu_phase_seconds'],'prior_training_seconds':prior_seconds})
    if step==0 and not args.resume:save('0000')
    training_window_start=time.perf_counter()
    if rank==0:write_json(args.out/'training_window.json',{'fresh_experiment_seconds':args.fresh_experiment_seconds,'started_epoch':time.time(),'already_spent_seconds':fresh_elapsed_before,'starting_step':step,'model_loading_excluded':True,'fresh_initialization':args.resume is None and args.init_from is None})
    budget=args.hours*3600;reason='completed';previous_seconds=0.
    try:
        while step<args.steps:
            if args.video_training_budget_seconds is not None and video_update_seconds>=args.video_training_budget_seconds:reason='video_training_budget_completed';break
            if fresh_window_finished(args.fresh_experiment_seconds,fresh_elapsed()):reason='fresh_hour_completed';break
            if any_rank(stopping['requested'] or (args.out/'STOP').exists(),device,world):reason='stopped';break
            spent=elapsed()+index['gpu_phase_seconds']+prior_seconds
            # Reserve for a full step; a running CUDA operation cannot be neatly
            # checkpointed halfway through. Provider runtime limit remains separate.
            if any_rank(spent+max(previous_seconds*2,900)>budget,device,world):reason='budget_reached';break
            mode='ref' if rng.random()<ref_fraction else 't2v'
            batch=[sampler.take() for _ in range(world)];e=batch[rank]
            if args.purpose=='smoke':
                # Deterministic coverage of full short/long clips, ref packing and
                # portrait geometry. Never train on fictitiously approved audio.
                if args.phase=='images':
                    if (step*world+rank)%3==0:e=max(pool,key=lambda e:e['record'].get('caption_characters',0))
                    elif (step*world+rank)%3==1:e=next((e for e in pool if e['record']['height']>e['record']['width']),pool[0])
                    else:e=next((e for e in pool if e['record']['width']>e['record']['height']),pool[0])
                elif step%4==0:mode='t2v';e=min(pool,key=lambda e:e['record']['frames'])
                elif step%4==1:
                    mode='t2v';e=min(pool,key=lambda e:e['record']['frames'])
                elif step%4==2:mode='t2v';e=max(pool,key=lambda e:e['record']['frames'])
                else:
                    mode='t2v';portraits=[e for e in pool if e['record']['height']>e['record']['width']]
                    e=min(portraits or pool,key=lambda e:e['record']['frames'])
            mode=presentation_mode(mode,e,first_video_update=(args.phase=='videos' and step==0 and ref_fraction>0))
            r=e['record']
            if 'inner_checkpoint_min_frames' in profile:set_inner_checkpoints(model,r['frames']>=profile['inner_checkpoint_min_frames'])
            tic=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
            media=torch.load(args.cache/e['media'],map_location='cpu',weights_only=True)
            text=torch.load(args.cache/e['conditions'][mode],map_location='cpu',weights_only=True)
            if media['key']!=e['key'] or text['key']!=e['key']:raise ValueError('Cached payload key mismatch')
            references=media['condition_latents'] if mode=='ref' else []
            conditional=dict(text['conditional'],condition_latents=references);unconditional=dict(text['unconditional'],condition_latents=references)
            shape=tuple(media['video_latents'].shape);audio_frames=media['audio_latents'].shape[-1]
            pack=PackedCondition(conditional,shape,audio_frames,str(device));unpack=PackedCondition(unconditional,shape,audio_frames,str(device))
            u=max(1e-4,rng.random());tv,ta=paired_times(u,cfg['loss']['video_shift'],cfg['loss']['audio_shift'])
            vr,ar,data_v,data_a,ref_rows=noise_sample(media,pack.refs,tv,ta,noise_rng,str(device))
            optimizer.zero_grad(set_to_none=True)
            timings={}
            def sync_time():torch.cuda.synchronize(device);return time.perf_counter()
            t=sync_time()
            with torch.no_grad(),style_strength(model,0):base_v,base_a=predict(model,pack,vr,ar,tv,ta,ref_rows)
            timings['base_forward']=sync_time()-t
            # Actual zero-style parity, in the same no-grad arithmetic, before update 1.
            if step==0 and args.init_from is None and args.resume is None:
                with torch.no_grad(),style_strength(model,1):zero_v,zero_a=predict(model,pack,vr,ar,tv,ta,ref_rows)
                error=max((base_v-zero_v).abs().max().item(),(base_a-zero_a).abs().max().item())
                if not torch.isfinite(zero_v).all() or not torch.isfinite(zero_a).all() or error!=0:raise RuntimeError(f'Zero-initialized style changed base output: max abs {error}')
                write_json(args.out/f'zero_style_parity_rank{rank}.json',{'max_abs_error':error,'video_id':r['id'],'mode':mode,'passed':True});del zero_v,zero_a
            un_v=un_a=None;t=sync_time()
            if u>=cfg['loss']['guidance_base_u_min']:
                with torch.no_grad(),style_strength(model,1):un_v,un_a=predict(model,unpack,vr,ar,tv,ta,ref_rows)
            target_v=guidance_target(data_v,un_v,cfg['loss']['guidance_scale_video']);target_a=guidance_target(data_a,un_a,cfg['loss']['guidance_scale_audio'])
            timings['unconditional_forward']=sync_time()-t
            mask_v=media['video_valid_frames'].repeat_interleave((shape[2]//2)*(shape[3]//2));mask_a=media['audio_valid_rows']
            t=sync_time()
            context=torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
            # Keep style active AND the student's layout fixed through backward/recompute.
            with style_strength(model,1),context:
                pred_v,pred_a=predict(model,pack,vr,ar,tv,ta,ref_rows)
                loss,terms=style_loss(pred_v,pred_a,target_v,target_a,base_v,base_a,mask_v,mask_a,r['audio_supervised'],cfg['loss'])
            timings['student_forward']=sync_time()-t
            if any_rank(not torch.isfinite(loss),device,world):raise FloatingPointError('Non-finite loss on at least one replica')
            t=sync_time();loss.backward();timings['backward']=sync_time()-t
            missing=any(p.grad is None for p in params)
            finite=False if missing else bool(torch.stack([torch.isfinite(p.grad).all() for p in params]).all())
            if any_rank(not finite,device,world):raise FloatingPointError('Missing/nonfinite LoRA gradient on a replica')
            t=sync_time();average_adapter_gradients(params,world);timings['gradient_sync']=sync_time()-t
            grad=torch.nn.utils.clip_grad_norm_(params,optcfg['grad_clip'],error_if_nonfinite=True)
            if step in (0,2):
                sample=style_modules(model);selected=sample[:4]+sample[-4:]
                norms={n:{'A':0 if m.style_A.grad is None else float(m.style_A.grad.norm()),'B':0 if m.style_B.grad is None else float(m.style_B.grad.norm())} for n,m in selected}
                write_json(args.out/f'gradient_audit_{step+1:04d}_rank{rank}.json',norms)
                if any(v['B']==0 for v in norms.values()) or (step>=2 and any(v['A']==0 for v in norms.values())):raise RuntimeError('Missing early/late style gradient; inspect gradient audit')
            lr=optcfg['learning_rate']*min(1.,(global_offset+step+1)/optcfg['warmup_updates'])
            for group in optimizer.param_groups:group['lr']=lr
            t=sync_time();optimizer.step();timings['optimizer']=sync_time()-t;step+=1
            previous_seconds=sync_time()-tic
            if args.phase=='videos':video_update_seconds+=previous_seconds
            event={'training_phase':args.phase,'source_video_id':r.get('source_video_id',r['id']),'frame_epochs_completed':sampler.cycles+int(sampler.cursor==len(sampler.order)),'global_update_count':global_offset+step,'inner_checkpoint_enabled':r['frames']>=profile.get('inner_checkpoint_min_frames',0),'step':step,'id':r['id'],'mode':mode,'u':u,'loss':float(loss.detach()),'terms':{k:float(v.detach()) for k,v in terms.items()},'gradient_norm':float(grad),'lr':lr,'seconds':previous_seconds,'timings':timings,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,'elapsed_training_seconds':elapsed(),'cache_plus_training_seconds':elapsed()+index['gpu_phase_seconds']+prior_seconds}
            event['video_update_seconds']=video_update_seconds;event['audio_supervised']=r['audio_supervised'];event['reference_count']=len(references)
            event['rank']=rank;event['world_size']=world;event['presentations_completed']=step*world
            replica_events=gather(event,world)
            if rank==0:
                event['replicas']=replica_events
                with (args.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
                write_json(progress_path,dict(event,status='running'));print(json.dumps(event),flush=True)
            # Discard the entire step graph before saving or starting another sample.
            del loss,terms,pred_v,pred_a,base_v,base_a,target_v,target_a,data_v,data_a,un_v,un_a,pack,unpack,vr,ar,ref_rows,media,text
            if args.phase=='videos' and not numbered_checkpoint_due(step,args.checkpoint_every):save('last')
            if args.fresh_experiment_seconds is not None:
                if numbered_checkpoint_due(step,args.checkpoint_every):save(f'{step:04d}')
            elif step in cfg['pilot']['save_updates']:save(f'{step:04d}')
            elif numbered_checkpoint_due(step,args.checkpoint_every):save(f'{step:04d}')
    except BaseException as exc:
        # Any failure (including a partially executed optimizer.step) must not
        # overwrite the last atomic committed checkpoint. Graceful SIGTERM exits
        # through the ordinary loop boundary and saves normally below.
        write_json(args.out/f'failure_rank{rank}.json',{'status':'failed','step':step,'error_type':type(exc).__name__,'elapsed_training_seconds':elapsed()});raise
    save('last')
    if rank==0:write_json(progress_path,{'status':reason,'step':step,'elapsed_training_seconds':elapsed(),'cache_plus_training_seconds':elapsed()+index['gpu_phase_seconds']+prior_seconds,'world_size':world,'presentations_completed':step*world,'fresh_window_elapsed_seconds':fresh_elapsed(),'video_update_seconds':video_update_seconds,'next':'inspect matched-seed 8-NFE video validation before extending training'})
if __name__=='__main__':main()
