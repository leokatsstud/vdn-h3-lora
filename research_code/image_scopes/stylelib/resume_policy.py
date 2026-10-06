"""Allow only an explicit LR change; data, math, profile and RNG identity stay fixed."""
from .common import fingerprint

LEGACY_TRAINER_FINGERPRINT='f017cb0dd74652e720f3393f14951fde336288660e8113d812263890b38470e1'

def authorize_resume(saved,current,allow_learning_rate_change=False):
    new_identity=fingerprint(current)
    if saved['identity']==new_identity:return None
    if not allow_learning_rate_change:raise ValueError('Resume run identity differs')
    previous=saved.get('resume_identity_inputs')
    if previous is not None:
        if fingerprint(previous)!=saved['identity']:raise ValueError('Saved identity inputs do not match checkpoint')
        old_base={k:v for k,v in previous.items() if k!='effective_learning_rate'}
        new_base={k:v for k,v in current.items() if k!='effective_learning_rate'}
        if old_base!=new_base:raise ValueError('LR migration cannot change data, cache, code, precision, seed or purpose')
        old_rate=previous['effective_learning_rate']
    else:
        previous={k:v for k,v in current.items() if k!='effective_learning_rate'}
        previous['trainer_code']=LEGACY_TRAINER_FINGERPRINT
        if fingerprint(previous)!=saved['identity']:raise ValueError('Legacy checkpoint is not the exact pinned compatible run')
        if saved['config']!=current['contract']:raise ValueError('Legacy contract differs')
        old_rate=saved['config']['optimizer']['learning_rate']
    return {'old_identity':saved['identity'],'new_identity':new_identity,'step':saved['step'],
            'old_learning_rate':old_rate,'new_learning_rate':current['effective_learning_rate'],
            'allowed_change':'explicit_learning_rate_only; pinned legacy trainer compatibility upgrade',
            'optimizer_and_rng_preserved':True}
