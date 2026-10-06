# LoRA for VDN-H3.

> **Check the model license before using it.** The MiniMax H3 Community License excludes the EU, USA, UK and Republic of Korea from its applicable territory. Working with the open weights and derivative models in these regions requires separate permission from MiniMax, which can be requested through the form linked below. Official sources: [license text](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), [FAQ](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md), [authorization request](https://platform.minimax.io/h3-license). More on [licensing](docs/01-licensing.md).

Research into training artistic-style LoRA for **VDN-H3 `stage-dmd-step-250`**, an accelerated version of MiniMax H3. The goal is expressive 3D animation with stable figures and motion. VDN was chosen for its much higher generation speed than base H3, which is effectively impractical—or, more precisely, not cost-effective—even on H200/B200-class accelerators because of its expensive full attention. In most variants, replacing it with Sparse Attention only gives a 2–3× speedup across the full 50 denoising steps, still leaving generation at 5–15 minutes on an H200. In my opinion, if VDN loses any generation quality compared with original H3, the difference is small.

## Main results

Training on individual images learns the style quickly, but degrades motion in video. In every recipe we tested, direct training on VDN caused substantially more degradation than training on original MiniMax H3 and then transferring the LoRA to VDN. Transfer still leaves errors, but far fewer of them. The main problems are fast limb movements, interactions with objects and the behavior of complex mechanisms.

Continuing the directly trained VDN checkpoint on 60 video presentations did not convincingly restore motion, or even show a hint of recovery. Style is also unstable on new scenes: some outputs are fully photorealistic even though the same adapter strongly stylizes other scenarios. The causes of both effects remain unknown.

## Contents

- [Video comparisons](media/README.md).
- [Models and LoRA transfer](docs/02-model-and-transfer.md).
- [Training method](docs/03-training-method.md).
- [Experiments and results](docs/04-experiments.md).
- [Recipe comparison with Musubi](docs/10-recipes-and-controls.md).
- [FP8, LoRA application and motion degradation](docs/09-fp8-transfer-and-motion.md).
- [Data and short clips](docs/07-data-plan.md).
- [Open questions](docs/05-hypotheses.md).
- [Code](research_code/README.md).

## Next direction

Training original H3 on short videos in the target style, then transferring the adapter to VDN. We have prepared 513 captioned clips. On an H200, training on clips roughly 5–15 seconds long averages around 30 clips per hour, so we decided to test a dataset made mostly of very short clips: the majority are 39–124 frames, with a small portion of longer videos up to 15 seconds. This may speed up style learning. On 5–15-second clips, the first 60 presentations in the early video-only VDN LoRA run, without prior image training (that run later reached 181 updates), produced absolutely no change in photorealism apart from slight darkening across the shots. An effective number of video presentations is therefore probably above 1,000. A separate task is to find why direct VDN training causes stronger degradation and why style disappears on individual scenes. Initial image training on base H3 followed by video training might also give the best balance of quality and training speed, but that is only a hypothesis.

## Project license

Original materials: [MIT](LICENSE). Third-party code and the model retain their own licenses: [details](licenses/README.md).
