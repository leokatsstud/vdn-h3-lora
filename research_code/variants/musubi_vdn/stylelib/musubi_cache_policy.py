"""Reuse only exactly pinned image encoding; optimizer/loss are not cache math."""
from .common import fingerprint
def select_cache_contract(current,settings,phase,legacy_video_contract=None,original_contract=None):
    if phase!='images':raise ValueError('This package is image-only')
    if original_contract is None or fingerprint(original_contract)!=settings['contract_sha256']:raise ValueError('Unknown source encoding contract')
    for key in ['github_revision','hf_revision','checkpoint','media','captioning']:
        if current[key]!=original_contract[key]:raise ValueError('Image encoder contract changed: '+key)
    return original_contract
