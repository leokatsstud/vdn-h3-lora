# Models and LoRA transfer

Original MiniMax H3 generates high-quality video, but its dense attention takes considerable time. OpenVDN replaces it with hybrid attention and uses eight-step generation. The inference model in every transfer comparison is VDN-H3.

## Model version

| Component | Version |
|---|---|
| OpenVDN | `2a054b146d49c70ae89765c6f9f9593617b7e54d` |
| OpenVDN weights | `50744d80bde8c1adbc5f37089f125e0e436bf7b0` |
| Checkpoint | `stage-dmd-step-250` |
| Frozen base adapters | `default` and `turbo` |
| Diffusers | `3a2f35d4efa4c059c8bfb3bc0d6c906264895c81` |
| Text encoder | `42ed227ee7df40d41602854ae760620d6eb651fe` |

## Architectural differences

VDN combines local Softmax attention with linear memory across video frames. Both branches use the Q/K/V projections. Compatible dimensions allow H3 LoRA matrices to transfer, but the subsequent attention computation differs.

Original H3 already uses CFG-distillation ([MiniMax description](https://www.minimax.io/news/minimax-h3-open-source)). OpenVDN additionally trains the hybrid branch and performs few-step DMD for the `turbo` adapter. [OpenVDN code and description](https://github.com/OpenVDN/vdn-minimax-h3).

During style training, the base weights and built-in adapters are frozen. Only the new LoRA matrices are updated.

## Transfer result

A LoRA trained on original H3 transfers style to VDN and degrades motion less than the tested direct-VDN recipes. In E13, both training bases used BF16, identical adapter initialization, the first 500 frames, noise levels and learning rate. The difference remained. Its cause is unknown; the control does not separate hybrid attention from additional distillation. [Results](09-fp8-transfer-and-motion.md).

## Clip length and attention

In this codec, 22 source frames produce 7 latent frames; 39 produce 12; 56 produce 17; 124 produce 37; and 345 produce 102. With `chunk=5, radius=1`, a sequence of 7 latent frames is fully covered by the local window and bypasses the linear branch. A sequence of 12 is no longer fully covered.

Sources: [window geometry](https://github.com/OpenVDN/vdn-minimax-h3/blob/2a054b146d49c70ae89765c6f9f9593617b7e54d/src/models/softmax_attention/window.py), [hybrid attention](https://github.com/OpenVDN/vdn-minimax-h3/blob/2a054b146d49c70ae89765c6f9f9593617b7e54d/src/models/hybrid_attention.py).
