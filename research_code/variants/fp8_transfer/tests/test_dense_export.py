import copy,tempfile,unittest,torch
from pathlib import Path
from test_mlp_scope import Expanded
from stylelib.adapter import install_style,style_modules,export_style

class DenseExport(unittest.TestCase):
    def test_dense_keys_merge_into_hybrid_targets_with_same_delta(self):
        from src.inference.utils.lora import load_external_lora,merge_lora_state
        torch.manual_seed(91);dense=Expanded();hybrid=copy.deepcopy(dense)
        for block in dense.transformer_blocks:block.attn=block.attn.orig
        install_style(dense,4,4,12,896)
        with torch.no_grad():
            for _,m in style_modules(dense):m.style_B.normal_(0,.01)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'dense.safetensors';export_style(dense,p,{'alpha':4})
            state,scale,ranks=load_external_lora(str(p));self.assertEqual(ranks,[4]);self.assertEqual(scale,1)
            self.assertFalse(any('.attn.orig.' in k for k in state))
            before={n:p.detach().clone() for n,p in hybrid.named_parameters()}
            self.assertEqual(merge_lora_state(hybrid,state,scale),12)
            params=dict(hybrid.named_parameters())
            for name,a in state.items():
                if '.lora_A.' not in name:continue
                target=name.split('.lora_A.')[0]+'.weight'
                if '.attn.' in target:target=target.replace('.attn.','.attn.orig.')
                delta=state[name.replace('.lora_A.','.lora_B.')].float()@a.float()
                torch.testing.assert_close(params[target]-before[target],delta.to(params[target].dtype),rtol=1e-4,atol=1e-7)
