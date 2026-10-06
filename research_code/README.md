# Training code

[Experiment-to-code mapping](EXPERIMENTS.md)

[`image_scopes/`](image_scopes/) contains the E10 image trainers: FF-only, FF+V/O and full scope with preservation 0.3.

## Dependencies

Linux/CUDA, PyTorch, safetensors, Pillow, PyAV and filelock. OpenVDN is pinned in [model-version.json](../configs/model-version.json), and Diffusers/text encoder in [`dependency_pins.json`](image_scopes/dependency_pins.json). Upstream belongs in `image_scopes/upstream`.

`build_manifest.py` creates a manifest; `prepare_style_cache.py` encodes VAE and Qwen3-VL inputs; `train_style_vdn.py` trains the adapter. The manifest and cache hold geometry, captions, encoding parameters and record identities.

## Running

From `image_scopes/`:

```bash
python train_style_vdn.py --recipe ff_only --phase images --profile h200 \
  --purpose pilot --manifest /data/manifest.json --cache /data/cache \
  --weights /data/weights --out /data/runs/ff_only \
  --steps 2000 --seed 20261002 --learning-rate 0.0001 \
  --reference-fraction 0 --checkpoint-every 500 --fresh-experiment-seconds 3600
```

`ff_vo` and `full_guarded` use separate outputs and fresh adapter initialization. Full settings: [training method](../docs/03-training-method.md).

## Checks and resuming

```bash
python run_cpu_checks.py --single-process
```

CPU tests cover sampling, packing, loss, adapters and resume state. Continuation uses FP32 masters, Adam, RNG and sampler position. BF16 LoRA export is for generation, not exact optimizer restoration.
