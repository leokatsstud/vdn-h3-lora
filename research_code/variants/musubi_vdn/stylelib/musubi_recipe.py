"""One fixed Musubi-guidance image recipe; no preservation variants."""
import copy
def apply_image_recipe(config,name):
    if name!='musubi_guidance':raise ValueError('Only the reviewed Musubi-guidance recipe is supported')
    cfg=copy.deepcopy(config);s=cfg['style'];loss=cfg['loss']
    if (s['rank'],s['alpha'],s['blocks'],s['expected_modules'],s['expected_parameters'])!=(32,32,50,300,149094400):raise ValueError('Shared-QKV recipe contract differs')
    if (loss['guidance_scale_video'],loss['guidance_base_u_min'],loss['video_preservation_weight'],loss['audio_preservation_weight'])!=(4,.15,0,0):raise ValueError('Guidance-only loss contract differs')
    if cfg['optimizer']['warmup_updates']!=50:raise ValueError('Warmup contract differs')
    return cfg
