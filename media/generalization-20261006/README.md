# E16 generation metadata

Each pair uses VDN with two adapters after 4000 presentations: `dense4000_to_vdn` was trained on original H3, and `direct_vdn4000` on VDN. The `_merged` suffix means BF16 merging before FP8 quantization.

JSON files next to the MP4s contain the full prompt, seed, adapter/initial-noise SHA, geometry and generation settings. [Video comparison](../README.md).
