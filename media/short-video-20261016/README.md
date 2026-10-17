# Image training versus short-video fine-tuning

An original MiniMax H3 LoRA trained on 4000 image presentations was continued on **495 short clips, 39–124 frames (1.625–5.167 seconds)**, with clip-specific captions and real audio. On one B200, the measured rate was approximately **140–150 clip presentations per hour**, excluding model loading, cache preparation and evaluation.

The aim was to improve motion while retaining the learned style. Instead, **after 890 video presentations the two combat scenes showed more degradation than the image-only checkpoint**. This run did not demonstrate motion recovery.

- **Left:** original H3 LoRA after 4000 image presentations, transferred to VDN.
- **Right:** the same LoRA after 890 additional short-video presentations on original H3, transferred to VDN.

Both columns use VDN DMD250, separate LoRA residuals at strength 1, 8 NFE, 1344×768, 345 frames at 24 fps (14.375 seconds), with no reference images. Each pair uses the same prompt and seed. The prompts contain no explicit animation or 3D style instructions; the style trigger is `vstyleh3`.

The video phase also changed guidance correction from 3 to 4, removed separate base-prediction preservation losses, enabled real-audio supervision at weight 1 and switched from the earlier FP8 training implementation to native Musubi with a BF16 base. The cause of the regression is unresolved; this comparison does not isolate clip length.

[Training details](../../docs/12-short-video-finetuning.md) · [Comparison metadata](../../evidence/short-video-comparison.json)

## C01 — Sword fighter versus a luminous beast

<table>
<thead><tr><th>4000 image presentations<br>H3 LoRA transferred to VDN</th><th>4000 images + 890 video presentations<br>H3 LoRA transferred to VDN</th></tr></thead>
<tbody><tr>
<td width="420"><video src="https://github.com/user-attachments/assets/77af540f-6435-41c3-aef1-bc1a833fbd6e" width="420" controls></video></td>
<td width="420"><video src="https://github.com/user-attachments/assets/1b036cbd-97c6-4478-9034-a1a99f35dabc" width="420" controls></video></td>
</tr></tbody>
</table>

<details>
<summary>Full prompt · seed 2026101601 · identical within the pair</summary>

<pre>vstyleh3. A continuous wide tracking shot in a rain-soaked ruined courtyard at night. A human sword fighter with short dark hair wears scratched steel shoulder plates over woven indigo cloth, a frayed red sash and scuffed brown leather boots. Opposite him is a large four-legged horned beast with overlapping charcoal-black scales, rough chipped ridges, translucent amber claws and thin glowing turquoise veins running beneath its cracked skin. Wet flagstones reflect the turquoise light and orange lanterns. The beast charges rapidly from the right, plants its front paws and sweeps a heavy claw across the fighter. He ducks, pivots on his left foot and parries with both hands gripping a long polished steel sword. The blade leaves a narrow white-blue light trail that follows its actual arc, briefly lighting his fingers and the beast&#x27;s ridged face. He runs three fast steps around its flank; the beast turns, kicks loose stones and lunges again. The fighter jumps sideways over its sweeping tail, lands with bent knees and delivers a rising sword strike against an armored foreleg. Sparks scatter at the point of contact while the beast recoils, opens its jaws and roars. Both remain clearly visible as the camera tracks around them, showing foot placement, changing balance and sustained physical contact. Water sprays from their steps, the sash whips through the air and the beast&#x27;s luminous veins pulse along its shoulders. Metallic impacts, scraping claws, splashing footsteps and a low roar accompany the action.</pre>

</details>

## C02 — Spear fighter versus an obsidian guardian

<table>
<thead><tr><th>4000 image presentations<br>H3 LoRA transferred to VDN</th><th>4000 images + 890 video presentations<br>H3 LoRA transferred to VDN</th></tr></thead>
<tbody><tr>
<td width="420"><video src="https://github.com/user-attachments/assets/1df19d7b-7974-4b29-8d86-c16822d6cc82" width="420" controls></video></td>
<td width="420"><video src="https://github.com/user-attachments/assets/e37c054e-7a6c-44e6-bb44-d33e398ef05a" width="420" controls></video></td>
</tr></tbody>
</table>

<details>
<summary>Full prompt · seed 2026101602 · identical within the pair</summary>

<pre>vstyleh3. A continuous moving medium-wide shot follows a human spear fighter battling a towering stone guardian across a broken bridge above a misty gorge at sunset. The fighter has tightly braided black hair, a sand-colored woven coat with visible stitching, dark flexible leather gloves, hammered bronze arm guards and dust-covered boots. Her spear has a dark wooden shaft wrapped with worn leather and a silver leaf-shaped blade etched with fine luminous gold lines. The guardian has a clearly defined head, two arms and two legs made of angular obsidian plates with glassy chipped edges, porous basalt joints and branching molten-orange fissures beneath the surface. Wind blows grit and loose fabric across the bridge. The guardian rushes forward and swings one massive arm horizontally. She slides beneath it, grips the shaft with both hands and thrusts into the gap beside its elbow. A thin golden streak follows the spear tip; bright fragments and sparks fly from the exact contact point. It pulls back, stamps one foot and strikes downward with both fists. She springs sideways, lands on the next intact slab and rapidly spins the spear behind her back before catching it with both hands. The guardian twists toward her and sweeps its leg; she vaults over the sweep using the spear, turns in the air and lands in a stable crouch. She advances with two quick alternating thrusts while the guardian blocks and stumbles backward, its joints flexing and orange fissures brightening. The camera travels alongside them, keeping both full bodies visible and maintaining the bridge geometry. Stone chips tumble into the gorge, fabric ripples, the spear bends slightly under load and the fighters continuously adjust their footing. Heavy stone impacts, rapid footsteps, a ringing spear blade and strong wind accompany the action.</pre>

</details>
