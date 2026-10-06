# Open questions

The main unresolved result: direct LoRA training on individual VDN frames disrupts motion far more than training on original H3 and transferring to the same VDN.

| Possible mechanism | Basis | Test |
|---|---|---|
| Adapting shared Q/K/V on frames changes long-sequence processing | VDN uses these projections in local and linear attention; the linear branch is inactive on one frame | Train on real short videos and control the adaptation scope |
| The objective does not sufficiently preserve the few-step trajectory | Preservation compares predictions on training inputs, not an entire long generation | Matched objective controls |
| The data provides insufficient motion supervision | Individual frames contain no pose transitions | Video-only training on clips with moving people and objects |
| FP8 and the application path change the adapter's effect | Merging before FP8 suppressed style in one control, but not on every scene | Separate and merged application on identical inputs |

FF-only and removing extra preservation losses did not eliminate degradation. E13 also retained the H3/VDN difference in BF16, so FP8 alone is not a sufficient explanation.

A separate question is style disappearing on some new scenes. Other scenarios are strongly stylized, yet the dance rehearsal is fully photorealistic. Prompt formatting and adapter application were checked; the cause has not been found.

Related work: [MotionDirector](https://arxiv.org/abs/2310.08465), [VideoCrafter2](https://arxiv.org/abs/2401.09047), [Musubi H3 advanced](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/minimax_h3_advanced.md).
