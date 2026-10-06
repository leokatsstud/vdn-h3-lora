def presentation_mode(requested,entry,first_video_update=False):
    eligible=bool(entry['record']['references']) and bool(entry['conditions'].get('ref'))
    if first_video_update and eligible:return 'ref'
    return requested if requested!='ref' or eligible else 't2v'

def put_first_reference_first(sampler,pool):
    position=next((j for j,i in enumerate(sampler.order) if pool[i]['record']['references'] and pool[i]['conditions'].get('ref')),None)
    if position is None:raise ValueError('No reference presentation available')
    sampler.order[0],sampler.order[position]=sampler.order[position],sampler.order[0]
