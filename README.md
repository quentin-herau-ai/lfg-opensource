# LFG Open Source

This repository contains the open-source local inference code path for LFG. It loads a trained `LFG` checkpoint and runs it on either a video file or an ordered sequence of RGB images.

This public version runs local inference only. It loads local checkpoint files and local video/image inputs; it does not include training or multi-view workflows.

All input and checkpoint arguments must be ordinary filesystem paths on the same machine. URI-style locations are rejected by the CLI.

## Installation

```bash
conda create -n lfg-infer python=3.10 -y
conda activate lfg-infer

# Install PyTorch for your CUDA/CPU platform first if needed:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt

# Optional: install the CLI entry point.
pip install -e .
```

By default the CLI uses `--device auto`, which selects CUDA when a local GPU is available and otherwise falls back to CPU. You can force a device with `--device cuda`, `--device cuda:0`, or `--device cpu`.

## Checkpoint

Place the single-view LFG checkpoint on the same machine, for example:

```bash
mkdir -p checkpoints
# put the provided local checkpoint at checkpoints/lfg.pt
# or, for motion outputs:
# put the provided local motion checkpoint at checkpoints/pzow1k_seg_motion.pt
```

The checkpoint must contain single-view `LFG` weights, usually under `model_state_dict`. Checkpoints are loaded with PyTorch's safe weights-only loader. If the checkpoint includes saved config metadata or `args`, `infer.py` reads `M`, `N`, architecture settings, and optional heads automatically from plain dictionaries, `argparse.Namespace`, `types.SimpleNamespace`, or YACS config metadata. It also infers decoder size, autoregressive layer count, point head type, and optional heads from state-dict keys when possible. `M + N` must be at most 15 frames. The loader refuses partial model loads so inference does not run with randomly initialized weights. If metadata is missing, pass the overrides shown below.

## Run On A Video

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg.pt \
  --output-dir outputs/video_demo
```

If installed with `pip install -e .`, the same command is available as:

```bash
lfg-infer /path/to/video.mp4 \
  --checkpoint checkpoints/lfg.pt \
  --output-dir outputs/video_demo
```

Useful video options:

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg.pt \
  --frame-stride 3 \
  --max-frames 120 \
  --window-stride 1 \
  --output-dir outputs/video_dense
```

## Run On Images

Directory input:

```bash
python infer.py /path/to/frames \
  --checkpoint checkpoints/lfg.pt \
  --output-dir outputs/frames_demo
```

Glob input:

```bash
python infer.py "/path/to/frames/*.jpg" \
  --checkpoint checkpoints/lfg.pt \
  --output-dir outputs/glob_demo
```

Image files are sorted with natural numeric ordering, so `frame_2.jpg` comes before `frame_10.jpg`.

## Checkpoint Overrides

Use these when a checkpoint does not carry enough config metadata:

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg.pt \
  --m 3 \
  --n 3 \
  --ar-n-heads 8 \
  --ar-n-layers 4 \
  --use-segmentation-head \
  --segmentation-classes 7 \
  --output-dir outputs/manual_config
```

Other overrides:

```bash
--encoder-name dinov2
--decoder-size large   # small, base, or large
--ar-dropout 0.1
--use-motion-head / --no-use-motion-head
--use-flow-head / --no-use-flow-head
--point-head-type linear   # linear, refined, conv, or simple_conv
--target-size 518   # must be divisible by 14
--resize-mode crop   # or pad
--keep-ratio
```

## Outputs

Each model window is written under:

```text
outputs/.../
  run_metadata.json
  window_000000/
    metadata.json
    predictions.npz
    depth/000.png
    confidence/000.png
    segmentation/000.png      # only when the checkpoint has a segmentation head
    motion/000.png            # only when the checkpoint has a motion head
    flow/000.png              # only when the checkpoint has a flow head
```

`predictions.npz` can contain:

| Key | Shape | Meaning |
|---|---:|---|
| `local_points` | `[M+N, H, W, 3]` | Per-frame local 3D point map. Depth is `[..., 2]`. |
| `points` | `[M+N, H, W, 3]` | Points transformed by predicted camera poses. |
| `conf` | `[M+N, H, W, 1]` | Confidence logits. |
| `camera_poses` | `[M+N, 4, 4]` | Predicted camera poses. |
| `segmentation` | `[M+N, H, W, C]` | Segmentation logits, if enabled. |
| `motion` | `[M+N, H, W, 1]` | Motion logits, if enabled. |
| `flow` | `[M+N, H, W, 2]` | Optical-flow logits, if enabled. |

For long videos or image sequences, inference streams sampled frames through sliding windows instead of decoding the full input into memory first. The first `M` predictions correspond to the input/history frames for that window; the next `N` predictions are autoregressive future predictions. The JSON metadata records the source frame indices and which slots are padded for short tail windows.

## Inspect Without Running Inference

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg.pt \
  --inspect-only \
  --output-dir outputs/inspect
```

This validates input discovery and checkpoint loading, then writes `run_metadata.json`. For videos, inspect mode uses container metadata when available and does not decode or store every frame.

`--inspect-only` does not construct the full model. It reports checkpoint metadata, inferred `M`/`N`, optional heads, and whether the checkpoint is compatible with this single-view inference repo.

## Development Checks

```bash
python -m py_compile infer.py lfg/*.py $(find Pi3 -type f -name '*.py' | sort)
python -m pip wheel . --no-deps -w /tmp/lfg_wheel
```

Before publishing a wheel, install it into a temporary target and check that the bundled Pi3 package resolves outside the source tree:

```bash
rm -rf /tmp/lfg_install_check
python -m pip install /tmp/lfg_wheel/lfg_opensource-*.whl --no-deps --target /tmp/lfg_install_check
PYTHONPATH=/tmp/lfg_install_check /tmp/lfg_install_check/bin/lfg-infer --help
PYTHONPATH=/tmp/lfg_install_check python -c "from lfg.model import build_model; from lfg.config import ModelConfig; build_model(ModelConfig(decoder_size='small', ar_n_layers=1, n=0))"
```

## Notes For Public Release

- Do not commit model checkpoints to the repository. Keep checkpoints as separate local files when running inference.
- The repository is Apache-2.0 licensed. Some bundled model components retain upstream Apache-2.0 headers; review `THIRD_PARTY_NOTICES.md`.
- This repo intentionally rejects multi-view checkpoints. Use a checkpoint trained for the single-view `LFG` architecture.
