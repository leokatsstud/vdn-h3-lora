import copy,json,tempfile,unittest
from pathlib import Path
import torch
from stylelib.media import media_geometry,validate_phase_records
from stylelib.balanced_sampler import SourceBalancedSampler
from stylelib.ledger import reconcile_events
from stylelib.stage_transition import validate_stage_transition
from stylelib.cache_policy import select_cache_contract
from stylelib.common import fingerprint
from stylelib.distributed import gather

class StagedWorkflow(unittest.TestCase):
    def test_still_and_video_geometry(self):
        base={'width':1344,'height':768,'fps':24,'audio_supervised':False}
        self.assertEqual(media_geometry(dict(base,media_kind='image',frames=1)),((24,1,48,84),2))
        self.assertEqual(media_geometry(dict(base,frames=124)),((24,37,48,84),207))
        self.assertEqual(media_geometry(dict(base,frames=345)),((24,102,48,84),575))
        for frames in (0,2,4,6,123):
            with self.assertRaises(ValueError):media_geometry(dict(base,frames=frames))
        with self.assertRaises(ValueError):validate_phase_records([dict(base,frames=124)],'images')
    def pool(self):
        return [{'record':{'id':f'{source}_{i}','source_video_id':source}} for source,n in [('A',3),('B',8),('C',1)] for i in range(n)]
    def test_source_coverage_and_frame_no_replacement(self):
        s=SourceBalancedSampler(self.pool(),17);seen={k:[] for k in ('A','B','C')}
        for _ in range(8):
            cycle=[s.take()['record'] for i in range(3)]
            self.assertEqual({r['source_video_id'] for r in cycle},{'A','B','C'})
            for r in cycle:seen[r['source_video_id']].append(r['id'])
        self.assertEqual(len(set(seen['B'])),8);self.assertEqual(len(set(seen['A'][:3])),3)
    def test_sampler_resume_exact(self):
        s=SourceBalancedSampler(self.pool(),31)
        for _ in range(5):s.take()
        state=copy.deepcopy(s.state_dict());expected=[s.take()['record']['id'] for _ in range(40)]
        other=SourceBalancedSampler(self.pool(),999);other.load_state_dict(state)
        self.assertEqual([other.take()['record']['id'] for _ in range(40)],expected)
        bad=copy.deepcopy(state);bad['frame_orders']['A']=[99]
        with self.assertRaises(ValueError):other.load_state_dict(bad)
    def test_ledger_recovery_preserves_uncommitted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.jsonl';p.write_text('\n'.join(json.dumps({'step':n}) for n in range(1,6))+'\n{partial')
            self.assertEqual(reconcile_events(p,3),3)
            self.assertEqual([json.loads(x)['step'] for x in p.read_text().splitlines()],[1,2,3])
            self.assertEqual(len(list(Path(d).glob('events_uncommitted_*.jsonl'))),1)
            self.assertEqual(reconcile_events(p,3),0)
    def test_ledger_rejects_committed_gap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.jsonl';p.write_text('{"step":2}\n')
            with self.assertRaises(ValueError):reconcile_events(p,2)
    def test_phase_initialization_requires_same_math_and_complete_optimizer(self):
        cfg={'style':{'rank':32},'loss':{'guidance':3}}
        info={'vdn_revision':'fixed','text_revision':'fixed','style_modules':300}
        saved={'training_phase':'images','purpose':'pilot','step':100,'config':cfg,'model_info':info,'optimizer':{'state':{i:{} for i in range(600)}},'identity':'images','global_update_count':100}
        self.assertEqual(validate_stage_transition(saved,cfg,info)['warmup_updates_offset'],100)
        for key,value in [('training_phase','videos'),('purpose','smoke')]:
            bad=copy.deepcopy(saved);bad[key]=value
            with self.assertRaises(ValueError):validate_stage_transition(bad,cfg,info)
        with self.assertRaises(ValueError):validate_stage_transition(saved,dict(cfg,loss={'guidance':4}),info)
    def test_video_cache_reuse_only_declared_legacy_contract(self):
        current={'style':{'rank':32},'loss':{'guidance':3}};old=copy.deepcopy(current);old['style']['rank']=64
        setting={'contract_sha256':fingerprint(old)}
        self.assertEqual(select_cache_contract(current,setting,'videos',old),old)
        with self.assertRaises(ValueError):select_cache_contract(current,setting,'images',old)
        bad=copy.deepcopy(old);bad['loss']['guidance']=4
        with self.assertRaises(ValueError):select_cache_contract(current,{'contract_sha256':fingerprint(bad)},'videos',bad)
    def test_single_replica_gather_does_not_create_circular_event(self):
        event={'step':1};event['replicas']=gather(event,1)
        self.assertEqual(json.loads(json.dumps(event))['replicas'],[{'step':1}])

class StillPacking(unittest.TestCase):
    def test_actual_native_layout_and_noise_for_one_frame(self):
        from test_core import layout_oracle
        layout_oracle()
        from stylelib.packing import PackedCondition,noise_sample,unpatchify
        condition={'prompt_embeds':torch.zeros(3,5120),'text_token_tags':torch.ones(3,dtype=torch.long)}
        pack=PackedCondition(condition,(24,1,4,6),2,'cpu')
        self.assertEqual(len(pack.vi),6);self.assertEqual(len(pack.ai),4)
        media={'video_latents':torch.ones(24,1,4,6),'audio_latents':torch.zeros(2,32,2)}
        v,a,target_v,target_a,refs=noise_sample(media,[],.25,.6,torch.Generator().manual_seed(3),'cpu')
        self.assertEqual(tuple(v.shape),(6,96));self.assertEqual(tuple(a.shape),(4,32));self.assertFalse(refs)
        self.assertEqual(tuple(unpatchify(target_v,(24,1,4,6)).shape),(1,24,1,4,6))
