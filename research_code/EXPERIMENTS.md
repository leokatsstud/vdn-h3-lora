# Experiment code

| Experiment | Implementation | Differences and reuse |
|---|---|---|
| E00 | `legacy_rank64` | Early trainer, rank/alpha 64, Q/K/V/O |
| E01–E02 | `video_rank32`, `video_rank32_fast` | Rank/alpha 32, attention + FF; continuation of the same adapter |
| E03 | `checkpoint_benchmark` | The same rank-32 trainer; offload and activation-checkpoint diagnostics without optimizer updates |
| E04 | `frames_balanced`, `frames` | Initial source-balanced sampling; dataset cleaning and transition to frame epochs |
| E05–E06 | `frames_hour`, `frames` | Shared image trainer; fresh initialization or continuation, different save intervals |
| E07 | `video_av` | Continuing an image adapter on videos; target audio and eligible references |
| E08–E09 | `strength_controls` | Shared FAST generator, different LoRA strengths and long/short prompts |
| E10 | `image_scopes` | FF-only, FF+V/O and full scope selected by recipe; shared training loop |
| E12 | `fp8_transfer` | Original H3 or VDN loading; FP8 weights and computation; separate LoRA and merging |
| E13 | `musubi_vdn`, `musubi_dense` | One training loop; guidance 4, shared QKV-A, no preservation; different architecture and configuration |
| E14 | `data_clips` | Scene cropping, frame/audio verification, caption context |
| E16 | `fp8_transfer` | One loaded VDN, two adapters; additional merged controls |

[Experiment parameters](experiment_map.json) · [Module versions](snapshot_manifest.json)

## Assembling a variant

```bash
python research_code/assemble_variant.py --variant fp8_transfer --out /tmp/h3-fp8
python research_code/assemble_variant.py --variant musubi_vdn --out /tmp/h3-musubi-vdn
python research_code/assemble_variant.py --variant musubi_dense --out /tmp/h3-musubi-dense
```

For E13: `train_images.py --architecture vdn` or `--architecture dense`, with the trainer's other arguments.

[Source provenance](../evidence/code-provenance.json).
