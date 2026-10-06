"""Image-to-video checkpoint transition."""
def validate_stage_transition(saved,current_contract,current_model):
    if saved.get('training_phase')!='images' or saved.get('purpose')!='pilot' or saved['step']<1:raise ValueError('Video initialization requires an actual image-stage pilot checkpoint')
    if saved['config']!=current_contract:raise ValueError('Phase transition cannot change base/loss/rank/alpha')
    for key in ('vdn_revision','text_revision','style_modules'):
        if saved['model_info'].get(key)!=current_model.get(key):raise ValueError('Phase transition model differs')
    if len(saved['optimizer']['state'])!=2*current_model['style_modules']:raise ValueError('Incomplete image optimizer state')
    return {'source_identity':saved['identity'],'source_phase':'images','destination_phase':'videos','source_step':saved['step'],'optimizer_preserved':True,'style_master_preserved':True,'new_data_sampler_and_noise_rng':'Explicit new-phase seeds; same-phase resume keeps its exact RNG','warmup_updates_offset':saved.get('global_update_count',saved['step'])}
