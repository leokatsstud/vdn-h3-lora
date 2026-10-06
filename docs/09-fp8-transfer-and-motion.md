# FP8, LoRA application and motion degradation

## Transfer from original H3

An adapter trained on original MiniMax H3 for 4000 image presentations transfers style to VDN. Motion errors remain, especially during fast actions, but are far fewer than with every tested direct-VDN recipe.

Base weights are frozen in both cases. Degradation occurs with LoRA attached; the original model does not change.

## Separate application and merging

A separate adapter computes `W·x + B·A·x`. Merging writes `W + B·A` into the base weight. These expressions are mathematically equivalent at the same precision, but BF16 rounding and subsequent FP8 quantization change the result.

Merging before FP8 suppressed style in one early control. Switching to a separate BF16 residual restored style, but strong motion degradation from direct VDN LoRA remained. In the new N03 relay scene, transferred LoRA retains style after merging as well. Style loss from merging is therefore not universal.

Native FP8 modules and separate-LoRA wrappers also differ in available QKV/SwiGLU optimizations. Comparing the paths includes both quantization and computational implementation.

| Series | Adapter application |
|---|---|
| BF16 image-checkpoint evaluations | Separate A/B |
| Early FAST V097/V110 and strength tests | BF16 merge followed by FP8 |
| Corrected H3 → VDN transfer evaluations | Separate BF16 residual on an FP8 base |
| New scenes E16 | 12 separate-adapter outputs; two merged controls on N03 |

## Recipe and precision control: E13

Original H3 and VDN were trained from scratch on 701 images in BF16. Rank/alpha 32, shared Q/K/V A, guidance 4 with threshold 0.15, video MSE only, no preservation, LR 1e-4, warmup 50.

Initial FP32 LoRA matrices, the first 500 frames, noise levels and LR matched. All B matrices started at zero. [Audit](../evidence/musubi-dense-vdn-pair-audit.json).

At step 500, direct VDN again strongly disrupted motion; continuation stopped at 508. H3 completed 500 updates, then its adapter was transferred to VDN. Both short LoRA generations had identical prompts, seed, initial noise, geometry and settings. Transfer was less damaging.

This control excludes BF16/FP8 and initial sampling differences as sufficient explanations, but does not separate hybrid attention from additional distillation.

## Attention on individual frames

VDN's linear branch is inactive on one frame, but the trained Q/K/V are later used in that branch during long generation. This is a possible risk mechanism. Its contribution has not been measured separately; FF changes and the distilled trajectory may also affect motion.

## Memory and speed

H200, batch 1, rank 32, FP32 masters and Adam. Video measurements cover three updates without references or target audio.

| Mode | Peak allocated, GiB | Median step, s |
|---|---:|---:|
| VDN BF16, images | 69.11 | 0.700 |
| VDN FP8, images | 49.45 | 0.779 |
| Original H3 FP8, images | 47.06 | 0.764 |
| VDN BF16, short video | 92.44 | 54.05 |
| VDN FP8, short video | 76.29 | 49.58 |

FP8 reduced memory; image training did not become faster in this measurement. Video peak reserved was 90.41 GiB. Full video training fit on an 80 GB H100 has not been confirmed.

Matched recipes and differences from the community trainer are described in the [training matrix](10-recipes-and-controls.md). [New-scene videos](../media/README.md).
