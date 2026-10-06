"""FF-only, FF+V/O and full-scope image-training configurations."""
import copy

FF=['ff.net.0.proj','ff.net.2']
VO=['attn.orig.to_v','attn.orig.to_out.0']
RECIPES={
    'ff_only':{'targets':FF,'modules':100,'parameters':86016000,'preservation':0.1},
    'ff_vo':{'targets':VO+FF,'modules':200,'parameters':126156800,'preservation':0.1},
    'full_guarded':{'targets':['attn.orig.to_q','attn.orig.to_k']+VO+FF,'modules':300,'parameters':166297600,'preservation':0.3},
}

def apply_image_recipe(config,name):
    if name not in RECIPES:raise ValueError('Unknown image recipe')
    style=config['style']
    if (style['rank'],style['alpha'],style['blocks'],style['expected_modules'],style['expected_parameters'])!=(32,32,50,300,166297600):
        raise ValueError('Recipe requires the exact validated full-scope rank32 base contract')
    cfg=copy.deepcopy(config);recipe=RECIPES[name]
    cfg['style'].update(target_suffixes=list(recipe['targets']),expected_modules=recipe['modules'],expected_parameters=recipe['parameters'])
    cfg['loss']['video_preservation_weight']=recipe['preservation']
    return cfg
