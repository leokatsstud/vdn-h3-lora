# New scenes for LoRA comparison

Active-scene descriptions for E16. [Results](../../media/README.md). Settings: VDN, 14.375 s, 1344×768, 24 fps, 8 NFE, strength 1, no references.

| ID | Scene | Test focus | Characters |
|---|---|---|---:|
| N01 | [Parcel relay in a railway concourse](N01.txt) | Running, stairs, handoff, crowds, emotion | 5731 |
| N02 | [Emergency tray handoff in a restaurant kitchen](N02.txt) | Hands, tray catch, body turn, multiple people | 5958 |
| N03 | [Relay through sports obstacles](N03.txt) | Fast running, jumping, balance, baton handoff; one shot | 5717 |
| N04 | [Fast partner rehearsal on stage](N04.txt) | Two couples, turns, jump, hand contact and gaze | 5804 |
| N05 | [Falling crates on a dock](N05.txt) | Multiple runners, object weight, catching and carrying | 5893 |
| N06 | [Fast bicycle change before a race](N06.txt) | Running, wheel rotation, tool work, changing support | 6195 |

Full prompts, seeds and SHA are in [suite.json](suite.json). N03's single long shot separates within-motion errors from cut-related problems. The other scenes contain three shots with sequential handoffs or changes in support.

The style trigger is `vstyleh3`. Dialogue is in `integrated_multimodal_description`.
