# Training recipes and comparison with Musubi

E01–E13 used an experimental trainer on OpenVDN/Diffusers. E13 reproduces the community Musubi guidance recipe, but does not run stock Musubi end to end.

| Recipe | Base and data | LoRA | Loss and optimization | Result |
|---|---|---|---|---|
| E01–E09 | VDN BF16; videos or frames | Independent Q/K/V/O and FF; rank/alpha 32; 166,297,600 parameters | Guidance 3, threshold 0.15; video/audio preservation; LR 1e-4, warmup 10 | Style appears on frames, motion degrades |
| E10 | VDN BF16; 2000 frames each | FF-only; FF+V/O; full scope | Full scope uses video preservation 0.3 | No scope eliminated degradation |
| E12 | Original H3 FP8; 780 frames, up to 4000 | Full scope; FP32 masters | Early guidance/preservation recipe | Style is retained on VDN; degradation is lower than with direct training |
| E13 | H3 and VDN BF16; 701 frames, 500 each | Shared Q/K/V A; rank/alpha 32; 149,094,400 parameters | Guidance 4, threshold 0.15; video MSE without preservation; LR 1e-4, warmup 50 | The motion-quality difference remained |
| E17 | Original H3 image4000 continued on 495 short videos; evaluation at 890 new video updates | Independent Q/K/V/O and FF; rank/alpha 32; retained image Adam states | Native Musubi BF16, guidance 4, threshold 0.15, real-audio weight 1, no separate preservation; LR 1e-4 | Paired VDN combat scenes degraded relative to image4000 |

## VDN adaptation versus recipe changes

VDN training requires a differentiable hybrid-attention forward. Inference-only kernels do not support backward. LoRA transfer requires matching QKV/FF and SwiGLU half ordering.

Preservation losses, guidance 3 instead of 4, warmup 10 and independent A matrices for Q/K/V are not required by VDN's architecture. These were separate choices in the early recipe. E13 removed the extra losses and brought guidance and QKV parameterization closer to Musubi; strong direct-VDN degradation remained.

Projection count, A/B matrix count and unique parameter count differ. With shared QKV-A, several export keys refer to one trainable tensor.

## Musubi

[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/minimax_h3.md) is a community trainer supporting H3. In the guidance approach used here, the target is adjusted through an unconditional prediction. Additional teacher-matching and training-adapter modes are described in the [advanced guide](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/minimax_h3_advanced.md); they were not used in these experiments.

Official third-party integration list: [MiniMax H3 training & LoRAs](https://github.com/MiniMax-AI/awesome-minimax-h3-integration#training--loras).

## Completed controls

- Base/zero-LoRA parity, finite gradients, Adam/RNG preservation.
- FF-only, FF+V/O and full scope with stronger preservation.
- Separate LoRA and merging before FP8.
- BF16 control with identical initialization and the first 500 inputs.
- FP8 on images and three short-video updates.
- Continuing an image-trained adapter on 60 video presentations with audio and references.

## Related work

[MotionDirector](https://arxiv.org/abs/2310.08465) studies separating appearance and motion; [VideoCrafter2](https://arxiv.org/abs/2401.09047) studies training video models on images and videos. Their recipes were not tested on VDN-H3 in this series.

[Short-video fine-tuning details and comparison](12-short-video-finetuning.md).
