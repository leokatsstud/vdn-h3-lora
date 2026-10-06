import copy,unittest
import torch
from torch import nn
from stylelib.image_recipe import apply_image_recipe,RECIPES
from stylelib.common import contract,fingerprint
from stylelib.adapter import install_style,style_modules,style_parameters
from stylelib.cache_policy import select_cache_contract
from test_mlp_scope import Expanded

def meta_model():
    # Use small tensors for scope and parameter-count checks.
    model=nn.Module();model.transformer_blocks=nn.ModuleList()
    for _ in range(50):
        b=nn.Module();b.attn=nn.Module();b.attn.orig=nn.Module()
        for n in ('to_q','to_k','to_v'):setattr(b.attn.orig,n,nn.Linear(5376,7168,device='meta'))
        b.attn.orig.to_out=nn.ModuleList([nn.Linear(7168,5376,device='meta')])
        b.ff=nn.Module();b.ff.net=nn.ModuleList([nn.Module(),nn.Identity(),nn.Linear(14336,5376,device='meta')]);b.ff.net[0].proj=nn.Linear(5376,28672,device='meta')
        model.transformer_blocks.append(b)
    return model

class ImageRecipes(unittest.TestCase):
    def test_trainer_media_contract_accepts_explicit_audio_argument(self):
        from stylelib.media import validate_phase_records
        image={'media_kind':'image','frames':1,'width':1344,'height':768,'audio_supervised':False,'audio_present':False}
        validate_phase_records([image],'images',False)
        video=dict(image,media_kind='video',frames=124,audio_supervised=True,audio_present=True)
        validate_phase_records([video],'videos',True)
        with self.assertRaises(ValueError):validate_phase_records([video],'videos',False)
    def test_common_FF_masters_start_identically_across_scopes(self):
        torch.manual_seed(91);full=Expanded();install_style(full,4,4,12,896)
        reference={n:m.style_A.detach().clone() for n,m in style_modules(full) if '.ff.' in n}
        torch.manual_seed(91);ff=Expanded();install_style(ff,4,4,4,384,target_suffixes=RECIPES['ff_only']['targets'])
        self.assertTrue(all(torch.equal(m.style_A,reference[n]) for n,m in style_modules(ff)))
    def test_exact_shapes_and_trainable_counts_without_allocating_weights(self):
        for name in RECIPES:
            cfg=apply_image_recipe(contract(),name);m=meta_model();style=cfg['style']
            paths=install_style(m,32,32,style['expected_modules'],style['expected_parameters'],target_suffixes=style['target_suffixes'])
            self.assertEqual(len(paths),style['expected_modules'])
            self.assertEqual(sum(p.numel() for p in style_parameters(m)),style['expected_parameters'])
            if name=='ff_only':self.assertTrue(all('.ff.' in p for p in paths))
            if name=='ff_vo':self.assertTrue(all('.to_q' not in p and '.to_k' not in p for p in paths))

    def test_ff_only_zero_parity_gradients_and_frozen_attention(self):
        torch.manual_seed(7);m=Expanded();x=torch.randn(3,8);before=m(x).detach()
        frozen={n:p.detach().clone() for n,p in m.named_parameters()}
        install_style(m,4,4,4,384,target_suffixes=RECIPES['ff_only']['targets'])
        torch.testing.assert_close(m(x),before,rtol=0,atol=0)
        opt=torch.optim.AdamW(style_parameters(m),lr=.01)
        for _ in range(3):
            opt.zero_grad();m(x).square().mean().backward();opt.step()
        self.assertTrue(all(a.style_A.grad.abs().sum()>0 and a.style_B.grad.abs().sum()>0 for _,a in style_modules(m)))
        self.assertTrue(all(torch.equal(m.get_parameter(n),p) for n,p in frozen.items()))
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if '.style_' not in n))

    def test_recipes_do_not_change_base_precision_or_original_contract(self):
        original=contract();snapshot=copy.deepcopy(original)
        for name in RECIPES:
            cfg=apply_image_recipe(original,name)
            for key in original:
                if key not in ('style','loss'):self.assertEqual(cfg[key],original[key])
        self.assertEqual(original,snapshot)

    def test_existing_cache_reused_only_for_declared_nonencoding_changes(self):
        original=contract();settings={'contract_sha256':fingerprint(original)}
        for name in RECIPES:
            cfg=apply_image_recipe(original,name)
            self.assertEqual(select_cache_contract(cfg,settings,'images',original_contract=original),original)
        bad=apply_image_recipe(original,'ff_only');bad['media']['fps']=12
        with self.assertRaises(ValueError):select_cache_contract(bad,settings,'images',original_contract=original)
        bad=apply_image_recipe(original,'ff_only');bad['loss']['video_shift']=8
        with self.assertRaises(ValueError):select_cache_contract(bad,settings,'images',original_contract=original)
