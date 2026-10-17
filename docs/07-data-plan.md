# Data and short clips

## Image sets

The initial frame set contained 846 images from 175 videos. The first selection retained 780 images, used in the main experiments. E13 uses a stricter selection of 701 images.

Each record contains an image and caption. Fresh runs use a random permutation: each frame exactly once per epoch.

## Video clips

513 clips were prepared from 174 source videos: 25 complete records and 488 crops within shots.

| Length | Clips | Proposed share of presentations |
|---|---:|---:|
| 39 frames | 313 | 50% |
| 56–124 frames | 182 | 45% |
| Over 124 frames | 18 | 5% |

Original geometry and 24 fps are preserved. No resizing, repeated-frame padding or joining different shots. Decoded frames and audio were compared with their source segments. Scenes shorter than 39 frames and some remaining tails were excluded.

The shares refer to presentations, not file counts: uniform sampling across all crops would sharply reduce the share of long videos. The proposed mixture including the 18 longer clips has not been tested; E17 used only the 495 clips at 39–124 frames.

## Captions

A caption describes its specific segment and contains three H3 fields: `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`. Speech and singing belong in the first field; ambient sounds in the second; music in the third.

## Length and latent representation

| Frames | Duration at 24 fps | Latent frames |
|---:|---:|---:|
| 22 | 0.92 s | 7 |
| 39 | 1.63 s | 12 |
| 56 | 2.33 s | 17 |
| 124 | 5.17 s | 37 |
| 345 | 14.38 s | 102 |

In the pinned VDN, 7 latent frames are fully covered by local attention, while 12 already require the linear branch. Short clips reduce token count. The 495 clips at 39–124 frames were subsequently used for video fine-tuning: approximately 140–150 presentations/hour on one B200. [Training result](12-short-video-finetuning.md).

Crops are encoded by the VAE again: slicing a full video's latents is not equivalent to encoding a new segment because of the encoder's causal context. [Dataset composition](../evidence/new-video-data-summary.json).

The short-video fine-tuning run used the 495 clips at 39–124 frames from this preparation, with their existing clip-specific captions and real audio; the 18 longer clips were not included.
