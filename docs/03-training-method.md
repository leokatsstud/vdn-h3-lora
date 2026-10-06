# What was trained

This is an experimental style trainer on top of VDN, not an official ready-made OpenVDN recipe.

The following describes the early full recipe used in E01–E09. For E10, the FP8 tests, the minimal guidance control E13 and Musubi guidance, see the [recipe matrix](10-recipes-and-controls.md). The earlier coefficients do not automatically apply to later experiments.

## Weights and optimizer

The full variant trained 300 linear projections across 50 blocks: the original attention Q/K/V/O and two FF projections. Rank and alpha are 32, for 166,297,600 trainable parameters. New A/B matrices are stored in FP32; LoRA is exported in BF16.

Base H3, the VDN branch, built-in `default`/`turbo`, encoders and VAE are **frozen**. Image training therefore did not damage the original weights. An attached adapter still changes the effective computation and can degrade motion. Disabling it restores the original effective VDN model.

The actual LR in the main experiments was **1e-4**, set explicitly. The 1e-5 value in the original contract does not describe those runs. AdamW: betas 0.9/0.999, eps 1e-8, weight decay 0, clipping 1, warmup 10 updates.

Training uses BF16, upstream FP32 sections, activation offload and differentiable FlexAttention. Fast inference-only kernels were not used for backward. A zero adapter is checked against the base; loss and gradients are checked for finite values.

## Loss

One base-noise value determines coupled video/audio times with shifts 12/3. The main target is adjusted for guidance:

```text
target = uncond_prediction + 3 * (flow_target - uncond_prediction)
loss = video_data_MSE
     + 0.1 * video_base_preservation_MSE
     + 0.1 * supervised_audio_MSE
     + 0.05 * audio_base_preservation_MSE
```

The extra unconditional forward is used at base u ≥ 0.15. It runs without gradients but with style-LoRA active. Preservation separately evaluates the base with style-LoRA disabled.

Audio terms apply only to valid audio rows. During image training, an audio-preservation coefficient alone does not guarantee that audio behavior is preserved. The `full_guarded` recipe increases video preservation to 0.3.

There is no separate geometry, optical-flow or temporal-consistency loss. Video training provides temporal supervision; preservation on one frame does not test a long sequence. The usefulness of this objective still needs comparison with alternatives.

## Data and counters

The first video run used 175 full native videos. Its first 100 updates covered 73 distinct sources, rather than a full pass without repeats.

The initial 846 frames were reduced to 780 from the same 175 sources by removing 66 images. Early source-balanced sampling did not show every frame exactly once. Fresh runs use a global random permutation: **all 780 frames exactly once per epoch**.

Video continuation from image checkpoint 5000 used 175 target videos with audio supervision enabled. Reference probability 0.5 applies to 23 eligible records, not the entire dataset. Under uniform selection, this is about 6.6% of presentations. An actual reference/audio update was separately checked on CUDA.

Batch size was 1 in the performed runs: one update equals one presentation.

## Resuming

The portable PT state stores FP32 masters, Adam, RNG and sampler position. BF16 LoRA does not replace it for exact continuation. Phase and sampler transitions preserve state while checking data and parameter identities.

Numbered checkpoints are saved atomically and are not overwritten.
