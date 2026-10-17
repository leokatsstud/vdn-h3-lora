# LoRA for VDN-H3.

> **Check the model license before using it.** The MiniMax H3 Community License excludes the EU, USA, UK and Republic of Korea from its applicable territory. Working with the open weights and derivative models in these regions requires separate permission from MiniMax, which can be requested through the form linked below. Official sources: [license text](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), [FAQ](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md), [authorization request](https://platform.minimax.io/h3-license). More on [licensing](docs/01-licensing.md).

Research into training artistic-style LoRA for **VDN-H3 `stage-dmd-step-250`**, an accelerated version of MiniMax H3. The goal is expressive 3D animation with stable figures and motion. VDN was chosen for its much higher generation speed than base H3, which is effectively impractical—or, more precisely, not cost-effective—even on H200/B200-class accelerators because of its expensive full attention. In most variants, replacing it with Sparse Attention only gives a 2–3× speedup across the full 50 denoising steps, still leaving generation at 5–15 minutes on an H200. In my opinion, if VDN loses any generation quality compared with original H3, the difference is small.

## Main results

Training on individual images learns the style quickly, but degrades motion in video. In every recipe we tested, direct training on VDN caused substantially more degradation than training on original MiniMax H3 and then transferring the LoRA to VDN. Transfer still leaves errors, but far fewer of them. The main problems are fast limb movements, interactions with objects and the behavior of complex mechanisms.

Continuing the directly trained VDN checkpoint on 60 video presentations did not convincingly restore motion, or even show a hint of recovery. Style is also unstable on new scenes: some outputs are fully photorealistic even though the same adapter strongly stylizes other scenarios. The causes of both effects remain unknown.

Continuing the original H3 image4000 LoRA on short videos also failed to improve motion: after **890 additional video presentations**, the matched combat scenes showed more degradation than the image-only checkpoint. Training on one B200 ran at approximately **140–150 short clips per hour**. [Experiment and paired videos](media/short-video-20261016/README.md).

## Contents

- [Video comparisons](media/README.md).
- [Models and LoRA transfer](docs/02-model-and-transfer.md).
- [Training method](docs/03-training-method.md).
- [Experiments and results](docs/04-experiments.md).
- [Recipe comparison with Musubi](docs/10-recipes-and-controls.md).
- [FP8, LoRA application and motion degradation](docs/09-fp8-transfer-and-motion.md).
- [Data and short clips](docs/07-data-plan.md).
- [Short-video fine-tuning results](docs/12-short-video-finetuning.md).
- [Open questions](docs/05-hypotheses.md).
- [Code](research_code/README.md).

## Next direction

Determine why video fine-tuning degraded the image-trained adapter. The next controls should separate changes in objective and audio supervision from clip duration, and check equivalence between the two training implementations.

## Project license

Original materials: [MIT](LICENSE). Third-party code and the model retain their own licenses: [details](licenses/README.md).
