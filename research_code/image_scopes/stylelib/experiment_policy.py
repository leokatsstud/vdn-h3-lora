"""Training-window limits and numbered checkpoint scheduling."""
def validate_fresh_seconds(seconds,phase,purpose,init_from):
    if seconds is None:return
    if phase!='images' or purpose!='pilot' or init_from is not None:raise ValueError('Fresh-hour experiment must be image-only, with no stage initialization')
    if not 0<float(seconds)<=3600:raise ValueError('Fresh image experiment may train at most3600seconds')

def fresh_window_finished(seconds,elapsed):
    return seconds is not None and elapsed>=seconds

def numbered_checkpoint_due(step,interval):
    return step>0 and step%interval==0
