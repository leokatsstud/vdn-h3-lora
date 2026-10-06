"""Native still/video geometry. A still is not a five-frame video."""
def latent_frames(frames):
    if frames==1:return 1
    if frames<5 or frames%17!=5:raise ValueError('Video frames must be17n+5; stills must be1')
    return (frames-5)//17*5+2

def media_geometry(record):
    w,h=record['width'],record['height'];frames=record['frames']
    if w<=0 or h<=0 or w%32 or h%32:raise ValueError('Native dimensions must be positive multiples of32')
    image=record.get('media_kind','video')=='image'
    if image!=(frames==1):raise ValueError('Image/video kind does not match frame count')
    if record.get('fps',24)!=24:raise ValueError('Pinned24fps timeline required')
    return (24,latent_frames(frames),h//16,w//16),2 if image else round(frames/24*40)

def validate_phase_records(records,phase):
    if phase not in ('images','videos'):raise ValueError('Unknown training phase')
    expected='image' if phase=='images' else 'video'
    for r in records:
        media_geometry(r)
        if r.get('media_kind','video')!=expected:raise ValueError('Do not silently mix training phases')
        if r['audio_supervised']:raise ValueError('This style-only workflow must not supervise source audio')
