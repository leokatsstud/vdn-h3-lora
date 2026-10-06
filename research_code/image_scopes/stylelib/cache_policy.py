from .common import fingerprint

def select_cache_contract(current,settings,phase,legacy_video_contract=None,original_contract=None):
    candidates=[current]
    if original_contract is not None:candidates.append(original_contract)
    if phase=='videos' and legacy_video_contract is not None:candidates.append(legacy_video_contract)
    for candidate in candidates:
        if fingerprint(candidate)!=settings['contract_sha256']:continue
        # LoRA scope and preservation strength do not change cached VAE/Qwen
        # tensors. Everything else stays pinned, and the trainer separately
        # verifies media/caption hashes and exact source revisions.
        def cache_equivalent(config):
            import copy
            value=copy.deepcopy(config);value.pop('style',None)
            if 'loss' in value:value['loss'].pop('video_preservation_weight',None)
            return value
        if cache_equivalent(current)!=cache_equivalent(candidate):raise ValueError('Cache contract differs beyond the declared scope/preservation experiments')
        return candidate
    raise ValueError('Unrecognized cache contract for this phase')
