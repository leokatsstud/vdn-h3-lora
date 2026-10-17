# Continuing an image-trained H3 LoRA on short videos

The starting point was the original H3 LoRA after 4000 image presentations. It was continued on 495 already captioned clips from 156 source videos, ranging from 39 to 124 frames at 24 fps (1.625–5.167 seconds). Both landscape and portrait clips participated, with their own captions and real audio.

## Training

One B200 processed approximately **140–150 clip presentations per hour** with the BF16 original H3 base, FP32 LoRA parameters and Adam states, without quantization or CPU offload. This is the measured rate for this short-clip mixture, which contains many 39-frame clips; it does not describe full-length video throughput. Model loading, cache preparation and evaluation are excluded.

The adapter retained its independent Q/K/V/O and FF factors: rank/alpha 32, 166,297,600 trainable parameters. Adam states from image step 4000 were retained; the video sampler and RNG started a new phase. Learning rate was 1e-4. The phase ended at 893 video updates; the paired evaluation uses checkpoint 890.

## Result

**After 890 video presentations, generation deteriorated instead of showing the expected improvement over image4000.** The two matched combat scenes show stronger degradation after video fine-tuning. Both adapters were evaluated on VDN DMD250 at strength 1 with identical prompts, seeds and settings within each pair.

[Paired videos](../media/short-video-20261016/README.md).

## Interpretation

Guidance correction was present on both stages: the target was rebuilt as `uncond + scale * (velocity - uncond)`, using a no-grad unconditional prediction with the LoRA active. The raw target was `clean_latent - noise`, with shared base-noise sampling and video/audio shifts 12/3.

The recipes were not otherwise identical. Image training used guidance 3, video base-prediction preservation 0.1 and audio preservation 0.05, without real-audio supervision. Video training used guidance 4 with the same base-sigma threshold 0.15, no separate preservation losses and real-audio loss weight 1. The frozen base also changed from FP8 in the earlier implementation to BF16 in native Musubi.

The regression therefore cannot be attributed specifically to short duration. Nor does it establish that the image checkpoint was trained incorrectly. The cause remains unresolved. Exact adapter conversion and state loading were checked; full prediction-and-gradient equivalence between the two training implementations has not been established.
