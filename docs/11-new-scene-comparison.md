# Comparison on new active scenes

## Conditions

Two adapters after 4000 presentations on 780 images: original H3 FP8 and direct VDN BF16 training. All outputs use the same VDN DMD250 with frozen `default`/`turbo`.

| Parameter | Value |
|---|---|
| LoRA strength | 1 |
| Reference | None |
| Resolution | 1344×768 |
| Length | 345 frames, 24 fps, 14.375 s |
| Generation | FAST FP8, 8 NFE, decomposed |
| Main pairs | Separate BF16 residual on an FP8 base |
| N03 control | BF16 merge, then FP8 |

Within each pair, prompt, seed and initial video/audio noise match. Adapter identities and application of all 300 target modules were checked. The text's style signal is `vstyleh3`; there are no 3D, CGI or animation instructions.

## Scenes

| ID | Scene | Actions |
|---|---|---|
| N01 | Railway concourse | Running, stairs, parcel handoff |
| N02 | Kitchen | Catching a tray, turning, hand contact |
| N03 | Relay | Running, jumping and baton handoff in one shot |
| N04 | Partner rehearsal | Two couples, turns, jumps, hand contact |
| N05 | Dock | Catching, carrying and lowering cargo |
| N06 | Bicycle | Tool work, running, mounting and pedaling |

[Videos and prompts](../media/generalization-20261006/README.md) · [Suite](../validation/generalization-20261006/suite.json) · [Metadata](../evidence/media-manifest.json).

## Results

The direct VDN adapter disrupts motion more than the transferred one. In N01, N03 and N05, the transferred adapter stylizes the scene more strongly. N02 shows stylization in both branches.

N04 is mostly photorealistic in both variants despite strong style from the same adapters elsewhere. The cause of missing style is unknown. Checks of prompts, adapter files and their application did not explain it.

In N03, transferred LoRA retains noticeable style both separately and after merging. Direct VDN LoRA is mostly photorealistic in both paths. Merging includes rounding, FP8 quantization and changes in available computational optimizations.

One seed per scene, without a new no-LoRA control. The two adapters differed in training base and precision. The reason transfer performs better has not been isolated.
