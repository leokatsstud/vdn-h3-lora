# Experiments and results

## Video training

| Experiment | Conditions | Result |
|---|---|---|
| E00 | Rank 64, repeated presentations of one video | Memorization of the character's appearance |
| E01 | 175 videos, rank 32, first 100 updates | No style in the first 60 updates; by 100, 73 distinct sources, with weak texture changes |
| E02 | Continued to 181 updates | No substantial gain in style |
| E07 | Images 5000 → videos 10/30/60; audio and references | 60 updates in 7247.04 s; no convincing recovery of motion |

## Image training

| Experiment | Conditions | Result |
|---|---|---|
| E04 | 100 presentations, then continued to 2210 | Style appears; motion degrades |
| E05 | From scratch, 780 images | 5037 updates in 3601.4 s; strong style, but quality does not improve monotonically |
| E06 | Continued 5037 → 10000 | Saved 7500/10000; further presentations neither fixed motion nor improved style |
| E08 | Checkpoints 3000/5000, strength 0.75/0.5/0.2 | Weak style on a new scene, even at 0.75 |
| E09 | Checkpoint 5000, strength 1; long and short prompts | Style visible at 1; motion more stable with the long prompt, but the action also differs |
| E10 | FF-only, FF+V/O, full scope with preservation 0.3; 2000 presentations each | All variants degrade motion; no convincing advantage for one scope |

In E05–E10, each frame is presented once per shuffled epoch. E04 used a different, source-balanced sampler.

## Transfer from original H3

| Experiment | Conditions | Result |
|---|---|---|
| E12 | Original H3 FP8, 780 images, up to 4000 presentations; inference on VDN | Style transfers; fast-motion degradation is substantially lower than with direct VDN training |
| E13 | H3 and VDN in BF16; 701 images, 500 presentations each; shared QKV, guidance 4, no preservation | Initialization and the first 500 inputs matched. Direct VDN again degraded motion strongly; H3 transfer was less damaging |
| E16 | Six new active scenes, two adapters after 4000 presentations; two extra merged-weight controls | The degradation difference remains. Both dance-rehearsal variants are photorealistic. Transfer retains style in the relay scene after merging |

[E16 videos](../media/README.md) · [Recipes](10-recipes-and-controls.md) · [Transfer analysis](09-fp8-transfer-and-motion.md).

## Speed and memory

E03: one forward/backward on checkpoint 100, seed 424242, base u=0.4.

| Memory policy | 124 frames | 345 frames |
|---|---:|---:|
| Inner checkpoints + offload | 73.74 s; 85.13 GiB | 209.70 s; 113.56 GiB |
| Block-only checkpoints + offload | 61.13 s; 92.23 GiB | 185.22 s; 133.01 GiB |

Loss matched, with relative gradient L2 difference 0. An additional long example hit OOM; block-only is therefore used up to 243 frames, and inner checkpoints from 260 frames onward.

E10: training time including saves, excluding model loading and generation.

| Adaptation scope | Presentations | Time |
|---|---:|---:|
| FF-only | 2000 | 1290.01 s |
| FF+V/O | 2000 | 1365.51 s |
| Full scope, preservation 0.3 | 2000 | 1431.10 s |

Early evaluations used BF16/Flex. FAST evaluations use `8nfe_tuned_fp8`, `decomposed`, 8 NFE. E05 includes 24 BF16/Flex generations: the base and 11 checkpoints, with and without a reference.

## Comparison limits

Generations cover a limited number of scenes and seeds. Early experiments differed in prompts, training dose and LoRA application path at the same time. E13 matched precision, initialization and the first 500 inputs, but did not separate attention from distillation. The reason transfer performs better remains unknown.
