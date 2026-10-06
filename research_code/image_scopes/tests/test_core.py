"""Small CPU proofs. Real VDN forward/backward remains a separate GPU gate."""
import ast,copy,json,math,sys,tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stylelib import GPU_ROOT
from stylelib.adapter import install_style,style_modules,style_parameters,style_strength,style_state,load_style_state,merge_style,export_style
from stylelib.packing import patchify,unpatchify,PackedCondition,noise_sample
from stylelib.objective import paired_times,guidance_target,style_loss
from stylelib.common import fingerprint,cache_key,atomic_torch_save
from stylelib.model import normalize_hybrid_storage
from stylelib.conditioning import condition_tokens,UNCONDITIONAL_PROBE
from stylelib.sampling import effective_reference_fraction

def layout_oracle():
    """Execute the ACTUAL pinned pure layout functions without importing CUDA-only deps."""
    source=(GPU_ROOT/'vendor_audit/before_denoise.py').read_text(encoding='utf-8')
    tree=ast.parse(source);nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):nodes.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id.startswith('_ROPE_') for n in node.targets):nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name in ('MiniMaxH3PrepareLayoutStep','MiniMaxH3Ref2VAPrepareLayoutStep'):
            methods=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name.startswith('build_')]
            nodes.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods,decorator_list=[]))
    env={'torch':torch,'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'pinned_layout','exec'),env)
    module=types.ModuleType('diffusers.modular_pipelines.minimax_h3.before_denoise')
    for n in ('MiniMaxH3PrepareLayoutStep','MiniMaxH3Ref2VAPrepareLayoutStep'):setattr(module,n,env[n])
    for package in ('diffusers','diffusers.modular_pipelines','diffusers.modular_pipelines.minimax_h3'):
        if package not in sys.modules:sys.modules[package]=types.ModuleType(package)
    sys.modules[module.__name__]=module
    return env

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.transformer_blocks=nn.ModuleList()
        for _ in range(2):
            b=nn.Module();b.attn=nn.Module();b.attn.orig=nn.Module()
            for name in ('to_q','to_k','to_v'):setattr(b.attn.orig,name,nn.Linear(8,12))
            b.attn.orig.to_out=nn.ModuleList([nn.Linear(12,8)]);b.ff=nn.Linear(8,8);self.transformer_blocks.append(b)
    def forward(self,x):
        for b in self.transformer_blocks:
            a=b.attn.orig;x=x+a.to_out[0](torch.tanh(a.to_q(x)+a.to_k(x)+a.to_v(x)));x=b.ff(x)
        return x

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.oracle=layout_oracle()
    def toy(self):
        torch.manual_seed(7);m=Toy();install_style(m,2,2,8,320);return m
    def test_new_hybrid_storage_preserves_alpha_fp32(self):
        m=self.toy();b=m.transformer_blocks[0];b.attn.linear_attention=nn.Module();b.attn.linear_attention.alpha=nn.Linear(8,3)
        b.attn.linear_attention.beta_proj=nn.Linear(8,2);b.attn.to_out_linear=nn.Linear(8,8)
        alpha=b.attn.linear_attention.alpha.weight.detach().clone()
        def predicate(n):return '.attn.' in n and '.attn.orig.' not in n
        normalize_hybrid_storage(m,predicate)
        self.assertEqual(b.attn.linear_attention.beta_proj.weight.dtype,torch.bfloat16)
        self.assertEqual(b.attn.linear_attention.alpha.weight.dtype,torch.float32)
        self.assertTrue(torch.equal(alpha,b.attn.linear_attention.alpha.weight))
        self.assertEqual(b.attn.orig.to_q.style_A.dtype,torch.float32)
        self.assertEqual(b.ff.weight.dtype,torch.float32)
    def test_zero_style_exact_and_frozen_only(self):
        torch.manual_seed(7);base=Toy();m=copy.deepcopy(base);install_style(m,2,2,8,320);x=torch.randn(3,8)
        self.assertTrue(torch.equal(base(x),m(x)))
        self.assertEqual(sum(p.numel() for p in style_parameters(m)),320)
        before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad};opt=torch.optim.AdamW(style_parameters(m),lr=.01)
        for _ in range(3):opt.zero_grad();m(x).square().mean().backward();opt.step()
        self.assertTrue(all(m.get_parameter(n).grad is None and torch.equal(p,m.get_parameter(n)) for n,p in before.items()))
        self.assertTrue(all(a.style_A.grad.norm()>0 and a.style_B.grad.norm()>0 for _,a in style_modules(m)))
        with style_strength(m,0):self.assertTrue(torch.equal(base(x),m(x)))
    def test_role_restoration_even_on_error(self):
        m=self.toy()
        with self.assertRaises(RuntimeError):
            with style_strength(m,0):raise RuntimeError('deliberate')
        self.assertTrue(all(a.strength==1 for _,a in style_modules(m)))
        self.assertTrue(all(p.requires_grad for p in style_parameters(m)))
    def test_resume_reproduces_next_optimizer_step(self):
        m=self.toy();x=torch.randn(5,8);opt=torch.optim.AdamW(style_parameters(m),lr=.01)
        def step(model,optimizer):optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        step(m,opt);state=style_state(m);ostate=copy.deepcopy(opt.state_dict());other=self.toy();load_style_state(other,state)
        oo=torch.optim.AdamW(style_parameters(other),lr=.01);oo.load_state_dict(ostate);step(m,opt);step(other,oo)
        self.assertTrue(all(torch.equal(v,style_state(other)[k]) for k,v in style_state(m).items()))
    def test_checkpoint_file_roundtrip_weights_only(self):
        import random
        m=self.toy();rng=random.Random(42);gen=torch.Generator().manual_seed(55)
        payload={'style':style_state(m),'sample_rng':rng.getstate(),'noise_rng':gen.get_state(),'step':3}
        with tempfile.TemporaryDirectory(dir=GPU_ROOT/'tests') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((GPU_ROOT/'tests').resolve()))
            path=Path(folder)/'last.pt';atomic_torch_save(payload,path)
            read=torch.load(path,weights_only=True,map_location='cpu')
            self.assertEqual(read['step'],3);other=self.toy();load_style_state(other,read['style'])
            rng2=random.Random();rng2.setstate(read['sample_rng']);self.assertEqual(rng.random(),rng2.random())
            gen2=torch.Generator();gen2.set_state(read['noise_rng']);self.assertTrue(torch.equal(torch.randn(5,generator=gen),torch.randn(5,generator=gen2)))
    def test_bf16_export_loads_in_actual_upstream_merger(self):
        from src.inference.utils.lora import load_external_lora,merge_lora_state
        m=self.toy()
        with torch.no_grad():
            for _,a in style_modules(m):a.style_B.normal_(0,.01)
        with tempfile.TemporaryDirectory(dir=GPU_ROOT/'tests') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((GPU_ROOT/'tests').resolve()))
            path=Path(folder)/'style.safetensors';export_style(m,path,{'alpha':2,'rank':2})
            state,scale,ranks=load_external_lora(str(path));self.assertEqual(scale,1);self.assertEqual(ranks,[2])
            self.assertTrue(all(v.dtype==torch.bfloat16 for v in state.values()))
            torch.manual_seed(7);base=Toy();pairs=merge_lora_state(base,state,scale);self.assertEqual(pairs,8)
            rounded={k.replace('.lora_A.weight','.style_A').replace('.lora_B.weight','.style_B'):v for k,v in state.items()}
            load_style_state(m,rounded);x=torch.randn(4,8);torch.testing.assert_close(base(x),m(x),rtol=3e-5,atol=3e-6)
    def test_export_merge_equivalence(self):
        m=self.toy();x=torch.randn(4,8)
        with torch.no_grad():
            for _,a in style_modules(m):a.style_B.normal_(0,.01)
        with style_strength(m,.5):expected=m(x)
        merge_style(m,.5);self.assertEqual(len(style_modules(m)),0)
        torch.testing.assert_close(m(x),expected,rtol=2e-5,atol=2e-6)
    def test_patching_matches_pinned_diffusers_and_inverse(self):
        for shape in [(1,24,2,4,6),(1,24,7,6,4)]:
            z=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)
            self.assertTrue(torch.equal(patchify(z),self.oracle['patchify_video_latents'](z,(1,2,2))))
            self.assertTrue(torch.equal(unpatchify(patchify(z),shape[1:]),z))
    def test_reference_and_unconditional_layouts(self):
        refs=[torch.ones(1,24,1,4,6),torch.ones(1,24,1,6,4)*2]
        def make(n):return PackedCondition({'prompt_embeds':torch.zeros(n,5120),'text_token_tags':torch.ones(n,dtype=torch.long),'condition_latents':refs},(24,7,4,6),3,'cpu')
        c=make(8);u=make(2);self.assertEqual(c.ncv,12);self.assertEqual(c.layout.video_start-u.layout.video_start,6)
        self.assertEqual(c.layout.num_frames,7);self.assertEqual(c.layout.tokens_per_frame,6);self.assertEqual(c.layout.frame_size,(2,3))
        media={'video_latents':torch.ones(24,7,4,6),'audio_latents':torch.ones(2,32,3)}
        vr,ar,tv,ta,rr=noise_sample(media,c.refs,.2,.4,torch.Generator().manual_seed(3),'cpu')
        ci=c.inputs(vr,ar,.2,.4,rr);ui=u.inputs(vr,ar,.2,.4,rr)
        self.assertTrue(torch.equal(ci['hidden_states'],ui['hidden_states']))
        row_times=ci['timestep'][ci['timestep_indices']]
        torch.testing.assert_close(row_times[c.vi[:c.ncv]],torch.full((12,),.999))
        torch.testing.assert_close(row_times[c.vi[c.ncv:]],torch.full((42,),.2))
        self.assertEqual(c.layout.video_start,int(c.vi[c.ncv]));self.assertEqual(c.layout.video_end,int(c.vi[-1])+1)
    def test_t2v_layout_and_noise_targets(self):
        p=PackedCondition({'prompt_embeds':torch.zeros(3,5120),'text_token_tags':torch.ones(3,dtype=torch.long)},(24,2,4,4),3,'cpu')
        self.assertEqual(p.ncv,0);self.assertEqual(p.layout.video_start,9)
        media={'video_latents':torch.ones(24,2,4,4),'audio_latents':torch.ones(2,32,3)}
        vr,ar,dv,da,_=noise_sample(media,[],.3,.5,torch.Generator().manual_seed(2),'cpu')
        torch.testing.assert_close(vr+.7*dv,torch.ones_like(vr));torch.testing.assert_close(ar+.5*da,torch.ones_like(ar))
    def test_guidance_and_masked_audio(self):
        p=torch.tensor([[1.,2.],[100.,100.]],requires_grad=True);a=torch.ones(2,2,requires_grad=True);z=torch.zeros_like(p);mask=torch.tensor([True,False])
        target=guidance_target(torch.ones(2,2),torch.full((2,2),.5),3);torch.testing.assert_close(target,torch.full((2,2),2.))
        cfg={'video_data_weight':1,'video_preservation_weight':.1,'clean_audio_data_weight':.1,'audio_preservation_weight':.05}
        loss,terms=style_loss(p,a,z,torch.full_like(a,999),z,z,mask,mask,False,cfg);loss.backward()
        self.assertEqual(float(terms['audio_data'].detach()),0);self.assertEqual(float(p.grad[1].abs().sum()),0);self.assertGreater(float(a.grad[0].abs().sum()),0)
        self.assertEqual(paired_times(1),(0.,0.));tv,ta=paired_times(.5);self.assertLess(tv,ta)
    def test_cache_invalidates_caption_and_reference_changes(self):
        r={'caption_sha256':'a','references':[{'sha256':'b'}]};k=cache_key(r,{'fps':24});r['references'][0]['sha256']='c';self.assertNotEqual(k,cache_key(r,{'fps':24}))
        r['caption_sha256']='d';self.assertNotEqual(k,cache_key(r,{'fps':24}))
    def test_text_only_never_invokes_image_processor(self):
        class Processor:
            def __init__(self):self.calls=[];self.tokenizer=self
            def __call__(self,text,add_special_tokens=False):self.calls.append(text);return {'input_ids':[220]}
            def image_processor(self,**kwargs):raise AssertionError('T2V must not call vision preprocessing')
        processor=Processor()
        ids,tags,vision=condition_tokens(processor,'A moving bird.',[])
        self.assertEqual(ids,[220]);self.assertEqual(tags,[1]);self.assertEqual(vision,{})
        condition_tokens(processor,UNCONDITIONAL_PROBE,[])
        self.assertEqual(processor.calls[-1],' ')
    def test_reference_condition_tokens_use_pinned_vision_builder(self):
        source=ast.parse((GPU_ROOT/'upstream/src/inference/encode_keyframes.py').read_text(encoding='utf-8'))
        func=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='build_presentation')
        env={'torch':torch,'TEXT_TAG':1,'VIDEO_TAG':0};exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),'pinned_presentation','exec'),env)
        class Tokenizer:
            def __call__(self,text,add_special_tokens=False):return {'input_ids':[len(text)]}
            def convert_tokens_to_ids(self,text):return {'<|vision_start|>':90,'<|image_pad|>':91,'<|vision_end|>':92}[text]
        class ImageProcessor:
            merge_size=2
            def __call__(self,images,return_tensors):
                self.asserted_images=len(images)
                return {'pixel_values':torch.zeros(2,4),'image_grid_thw':torch.tensor([[1,4,4],[1,4,4]])}
        proc=types.SimpleNamespace(tokenizer=Tokenizer(),image_processor=ImageProcessor())
        ids,tags,vision=condition_tokens(proc,' ',[object(),object()],reference_builder=env['build_presentation'])
        self.assertEqual(tags.count(0),12);self.assertEqual(tags.count(1),3);self.assertEqual(ids[-1],1)
        self.assertEqual(proc.image_processor.asserted_images,2);self.assertIn('image_grid_thw',vision)
    def test_reference_sampling_can_be_disabled(self):
        self.assertEqual(effective_reference_fraction(0,.15,0),0)
        self.assertEqual(effective_reference_fraction(None,.15,23),.15)
        self.assertEqual(effective_reference_fraction(.05,.15,23),.05)
        with self.assertRaises(ValueError):effective_reference_fraction(.1,.15,0)
    def test_clean_audio_term_is_presence_gated_and_weighted(self):
        v=torch.ones(2,2);a=torch.ones(2,2);z=torch.zeros(2,2);mask=torch.tensor([True,False])
        cfg={'video_data_weight':1,'video_preservation_weight':.1,'clean_audio_data_weight':.1,'audio_preservation_weight':.05}
        no,_=style_loss(v,a,z,z,z,z,mask,mask,False,cfg);yes,terms=style_loss(v,a,z,z,z,z,mask,mask,True,cfg)
        torch.testing.assert_close(yes-no,terms['audio_data']*.1)
    def test_unregistered_new_video_is_not_silently_ignored(self):
        import build_manifest
        with tempfile.TemporaryDirectory(dir=GPU_ROOT/'tests') as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to((GPU_ROOT/'tests').resolve()))
            (root/'_metadata').mkdir();(root/'_metadata/curated_manifest.json').write_text(json.dumps({'records':[]}),encoding='utf-8')
            (root/'V167.mov').write_bytes(b'placeholder')
            argv=sys.argv
            try:
                sys.argv=['build_manifest.py','--dataset',str(root),'--out',str(root/'out.json')]
                with self.assertRaisesRegex(ValueError,'New videos are not registered'):build_manifest.main()
            finally:sys.argv=argv

if __name__=='__main__':unittest.main()
