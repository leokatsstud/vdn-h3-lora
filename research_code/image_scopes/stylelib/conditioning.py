"""H3 text-only and image-reference presentations are distinct encoder paths."""
UNCONDITIONAL_PROBE=' '
TEXT_TAG=1

def condition_tokens(processor,prompt,images,reference_builder=None):
    if not images:
        ids=processor.tokenizer(prompt,add_special_tokens=False)['input_ids']
        if not ids:raise ValueError('Empty conditioning token sequence; do not strip the whitespace uncond probe')
        return ids,[TEXT_TAG]*len(ids),{}
    if reference_builder is None:
        from src.inference.encode_keyframes import build_presentation
        reference_builder=build_presentation
    return reference_builder(processor,prompt,images)
