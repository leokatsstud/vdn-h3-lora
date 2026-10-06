import ast,copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from stylelib.adapter import install_style,style_modules,style_parameters,style_state,load_style_state
from stylelib.musubi_objective import guidance_target,style_loss
from stylelib.musubi_recipe import apply_image_recipe
from stylelib.common import contract,fingerprint
from stylelib.frame_epoch_sampler import FrameEpochSampler
from stylelib.resume_policy import authorize_resume
from prepare_current_images import select_records,digest
from prepare_cache_subset import subset_index

def small_model():
    m=nn.Module();b=nn.Module();m.transformer_blocks=nn.ModuleList([b]);b.attn=nn.Module();b.attn.orig=nn.Module()
    for k in ['to_q','to_k','to_v']:setattr(b.attn.orig,k,nn.Linear(8,8))
    b.attn.orig.to_out=nn.ModuleList([nn.Linear(8,8)])
    b.ff=nn.Module();b.ff.net=nn.ModuleList([nn.Module(),nn.Identity(),nn.Linear(16,8)]);b.ff.net[0].proj=nn.Linear(8,16)
    install_style(m,4,4,6,384);return m

class TrainingMath(unittest.TestCase):
    def test_shared_qkv_forward_input_and_adapter_gradients_match_fused(self):
        torch.manual_seed(29);m=small_model();q,k,v=[getattr(m.transformer_blocks[0].attn.orig,n) for n in ['to_q','to_k','to_v']]
        self.assertIs(q.style_A,k.style_A);self.assertIs(q.style_A,v.style_A)
        for layer in [q,k,v]:layer.style_B.data.normal_()
        x=torch.randn(4,8,requires_grad=True);xf=x.detach().clone().requires_grad_(True)
        A=q.style_A.detach().clone().requires_grad_(True);B=torch.cat([l.style_B.detach() for l in [q,k,v]]).requires_grad_(True)
        out=torch.cat([l(x) for l in [q,k,v]],dim=-1)
        W=torch.cat([l.weight for l in [q,k,v]]);bias=torch.cat([l.bias for l in [q,k,v]])
        fused=torch.nn.functional.linear(xf,W,bias)+torch.nn.functional.linear(torch.nn.functional.linear(xf,A),B)
        out.square().mean().backward();fused.square().mean().backward()
        for a,b in [(out,fused),(x.grad,xf.grad),(q.style_A.grad,A.grad),(torch.cat([l.style_B.grad for l in [q,k,v]]),B.grad)]:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    def test_optimizer_has_no_duplicate_parameters_and_resume_keeps_shared_A(self):
        m=small_model();params=style_parameters(m);self.assertEqual(len(params),10);self.assertEqual(len(params),len({id(p) for p in params}))
        q=m.transformer_blocks[0].attn.orig.to_q;opt=torch.optim.AdamW(params,lr=.01)
        sum(layer(torch.randn(2,8)).square().mean() for _,layer in style_modules(m) if layer.in_features==8).backward();opt.step()
        state=style_state(m);other=small_model();load_style_state(other,state)
        self.assertEqual(len(state),10);self.assertIs(other.transformer_blocks[0].attn.orig.to_q.style_A,other.transformer_blocks[0].attn.orig.to_k.style_A)
        self.assertTrue(all(torch.equal(state[k],style_state(other)[k]) for k in state))
    def test_real_h3_shape_parameter_count_without_weights(self):
        m=nn.Module();m.transformer_blocks=nn.ModuleList()
        for _ in range(50):
            b=nn.Module();b.attn=nn.Module();b.attn.orig=nn.Module()
            for n in ['to_q','to_k','to_v']:setattr(b.attn.orig,n,nn.Linear(5376,7168,device='meta'))
            b.attn.orig.to_out=nn.ModuleList([nn.Linear(7168,5376,device='meta')]);b.ff=nn.Module();b.ff.net=nn.ModuleList([nn.Module(),nn.Identity(),nn.Linear(14336,5376,device='meta')]);b.ff.net[0].proj=nn.Linear(5376,28672,device='meta');m.transformer_blocks.append(b)
        install_style(m,32,32,300,149094400)
        self.assertEqual(len(style_parameters(m)),500);self.assertEqual(sum(p.numel() for p in style_parameters(m)),149094400)
    def test_guidance_target_matches_actual_pinned_musubi_statements(self):
        directory=os.environ.get('MUSUBI_TUNER_SOURCE')
        if not directory:self.skipTest('MUSUBI_TUNER_SOURCE is not configured')
        src=Path(directory)/'src/musubi_tuner/minimax_h3_train_network.py'
        tree=ast.parse(src.read_text(encoding='utf-8'))
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_apply_guidance_loss_targets')
        assignments=[n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['video_gap','video_target'] for t in n.targets)]
        self.assertEqual(len(assignments),2);code=compile(ast.Module(body=assignments,type_ignores=[]),str(src),'exec')
        for scale in [1.,3.,4.]:
            target=torch.randn(6,8);uncond=torch.randn(6,8);ns={'video_target':target.clone(),'uncond_video':uncond,'args':SimpleNamespace(h3_guidance_loss_scale=scale)}
            exec(code,ns);torch.testing.assert_close(guidance_target(target,uncond,scale),ns['video_target'],rtol=0,atol=0)
    def test_video_only_loss_has_no_audio_or_base_gradient(self):
        v=torch.randn(5,4,requires_grad=True);a=torch.randn(2,4,requires_grad=True);target=torch.randn_like(v,requires_grad=True)
        cfg=contract()['loss'];loss,_=style_loss(v,a,target,None,None,None,torch.ones(5,dtype=torch.bool),None,False,cfg)
        torch.testing.assert_close(loss,(v-target.detach()).square().mean());loss.backward()
        self.assertIsNone(a.grad);self.assertIsNone(target.grad)
    def test_extra_losses_rejected(self):
        cfg=copy.deepcopy(contract()['loss']);cfg['audio_preservation_weight']=.05
        with self.assertRaises(ValueError):style_loss(None,None,None,None,None,None,None,None,False,cfg)
    def test_old_resume_identity_rejected(self):
        current={'recipe':'musubi_guidance','shared_qkv':True};old={'recipe':'old','shared_qkv':False}
        with self.assertRaises(ValueError):authorize_resume({'identity':fingerprint(old),'resume_identity_inputs':old},current)
    def test_fixed_recipe(self):
        cfg=apply_image_recipe(contract(),'musubi_guidance');self.assertEqual(cfg['loss']['guidance_scale_video'],4);self.assertEqual(cfg['optimizer']['warmup_updates'],50)
        with self.assertRaises(ValueError):apply_image_recipe(cfg,'full_guarded')

class Selection(unittest.TestCase):
    def data(self,root):
        (root/'images').mkdir();(root/'captions').mkdir();rows=[]
        for ident in ['a','b','c']:
            image=root/'images'/f'{ident}.png';caption=root/'captions'/f'{ident}.txt';image.write_bytes(ident.encode());caption.write_text('Caption '+ident)
            rows.append({'id':ident,'video':f'images/{ident}.png','caption':f'captions/{ident}.txt','video_sha256':digest(image),'caption_sha256':digest(caption),'media_kind':'image','frames':1,'references':[],'audio_supervised':False})
        return {'records':rows}
    def test_deleted_images_excluded_even_with_orphan_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.data(root);(root/'images/b.png').unlink();kept,excluded=select_records(manifest,root)
            self.assertEqual([r['id'] for r in kept],['a','c']);self.assertEqual(excluded,['b'])
    def test_changed_retained_image_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.data(root);(root/'images/a.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):select_records(manifest,root)
    def test_filtered_cache_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a.pt').write_bytes(b'cached');record={'id':'a'};source={'complete':True,'records_sha256':'old','records':[{'record':record,'media':'a.pt','conditions':{'t2v':'a.pt'}},{'record':{'id':'deleted'},'media':'deleted.pt','conditions':{}}]};before=copy.deepcopy(source)
            selected=subset_index(source,{'records':[record],'records_sha256':'new'},root)
            self.assertEqual(source,before);self.assertEqual(len(selected['records']),1);self.assertEqual(selected['records_sha256'],'new')
    def test_epoch_covers_each_selected_image_once_and_resume(self):
        pool=[{'record':{'id':str(i),'source_video_id':'V1'}} for i in range(13)];s=FrameEpochSampler(pool,73)
        first=[s.take()['record']['id'] for _ in range(13)];self.assertEqual(set(first),{str(i) for i in range(13)});self.assertNotEqual(first,[str(i) for i in range(13)])
        state=s.state_dict();other=FrameEpochSampler(pool,73);other.load_state_dict(state)
        self.assertEqual([s.take()['record']['id'] for _ in range(13)],[other.take()['record']['id'] for _ in range(13)])
