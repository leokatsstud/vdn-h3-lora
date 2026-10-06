from .common import fingerprint

def select_cache_contract(current,settings,phase,legacy_video_contract=None):
    candidates=[current]
    if phase=='videos' and legacy_video_contract is not None:candidates.append(legacy_video_contract)
    for candidate in candidates:
        if fingerprint(candidate)!=settings['contract_sha256']:continue
        if {k:v for k,v in current.items() if k!='style'}!={k:v for k,v in candidate.items() if k!='style'}:raise ValueError('Cache contract differs beyond style scope')
        return candidate
    raise ValueError('Unrecognized cache contract for this phase')
