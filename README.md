<p align="center">
  <img src="assets/brand/icon.svg" width="96" alt="Applied Intuition"/>
</p>

<h1 align="center">LFG: Learning to Drive is a Free Gift — Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.22091"><img src="https://img.shields.io/badge/arXiv-2602.22091-b31b1b.svg" alt="arXiv"></a>
  <a href="https://lfg-ai.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/AppliedIntuitionResearch/LFG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Checkpoints-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

<p align="center">
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan <br>
  <b>CVPR 2026</b>
</p>

LFG learns a unified pseudo-4D representation — 3D point maps, camera poses, semantic layouts,
confidence and motion masks — from unposed, unlabelled dashcam video, supervised entirely by
frozen teacher models rather than human annotation. Given three observed frames it predicts all
of these for the observed frames *and* three future ones. This repository contains the local
inference path and the evaluation harness for the KITTI-360 and Waymo benchmarks.

## 🔥 News

- **[2026-08-12]** — Evaluation code released.
- **[2026-07-13]** — Checkpoint released on [Hugging Face](https://huggingface.co/AppliedIntuitionResearch/LFG).
- **[2026-06-14]** — Inference code released.
- **[2026-02-25]** — Paper on [arXiv](https://arxiv.org/abs/2602.22091); accepted at CVPR 2026.

## 📋 Table of Contents

- [Installation](#️-installation)
- [Checkpoints](#-checkpoints)
- [Getting Started](#-getting-started)
- [Outputs](#-outputs)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [License](#️-license)
- [Acknowledgments](#-acknowledgments)

## 🛠️ Installation

```bash
git clone https://github.com/Applied-Intuition-Open-Source/LFG.git
cd LFG
conda create -n lfg-infer python=3.10 -y
conda activate lfg-infer
pip install -r requirements.txt
pip install -e .          # optional, installs the lfg-infer entry point
```

**Requirements:** PyTorch 2.4+. A CUDA GPU is recommended but not required — the CLI falls back
to CPU. Install PyTorch for your platform first if the default wheel does not match your driver.

**External assets:** the checkpoint, from Hugging Face (below). Evaluation additionally needs
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) and the
[Waymo Open Dataset](https://waymo.com/open/), both of which require registration on their own
sites, and the baselines need their upstream packages — see [Evaluation](#-evaluation) for
both.

## 📦 Checkpoints

Pretrained weights are hosted on the [Applied Intuition Hugging Face organization](https://huggingface.co/AppliedIntuitionResearch/LFG).

| Model | Description | License | Download |
|---|---|---|---|
| `lfg_seg_motion_m3n3.pt` | 1.22B params. 3 observed frames in, 3 observed + 3 future out. Depth/points, camera pose, confidence, segmentation (7 classes), motion. | CC BY-NC 4.0 | [Hugging Face](https://huggingface.co/AppliedIntuitionResearch/LFG) |

```bash
pip install -U huggingface_hub
hf download AppliedIntuitionResearch/LFG lfg_seg_motion_m3n3.pt --local-dir checkpoints
```

## 🚀 Getting Started

### Run on a video

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --output-dir outputs/video_demo
```

If installed with `pip install -e .`, the same command is available as:

```bash
lfg-infer /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --output-dir outputs/video_demo
```

Useful video options:

```bash
python infer.py /path/to/video.mp4 \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --frame-stride 3 \
  --max-frames 120 \
  --window-stride 1 \
  --output-dir outputs/video_dense
```

### Run on images

Directory input:

```bash
python infer.py /path/to/frames \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --output-dir outputs/frames_demo
```

Glob input:

```bash
python infer.py "/path/to/frames/*.jpg" \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --output-dir outputs/glob_demo
```

Image files are sorted with natural numeric ordering, so `frame_2.jpg` comes before `frame_10.jpg`.

## 📤 Outputs

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

For long videos or image sequences, inference streams sampled frames through sliding windows
instead of decoding the full input into memory first. The first `M` predictions correspond to the
input/history frames for that window; the next `N` are autoregressive future predictions. The
JSON metadata records the source frame indices and which slots are padded for short tail
windows.

## 📊 Evaluation

`evaluate.py` scores depth, semantic segmentation and trajectory on KITTI-360 and the Waymo
Open Dataset. Each clip is six consecutive frames; LFG is given the first three and predicts all
six, so results are reported over all frames (`overall`) and over the three it had to predict
(`predicted`). Baselines that do not predict the future are given all six frames.

### Data

Two datasets are supported: [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) for depth,
segmentation and trajectory, and the [Waymo Open Dataset](https://waymo.com/open/) for depth and
trajectory. Both require registration on their respective sites.

#### KITTI-360

Follow the download instructions on the official site to obtain the perspective images,
Velodyne scans, calibrations, vehicle poses and 2D semantic labels, and unpack them into a
single dataset root. The shipped clip list covers sequences `2013_05_28_drive_0000_sync` and
`2013_05_28_drive_0002_sync` (~50 GB).

```text
KITTI-360/
  calibration/
  data_2d_raw/<sequence>/image_00/data_rect/*.png
  data_2d_semantics/train/<sequence>/image_00/semantic/*.png
  data_3d_raw/<sequence>/velodyne_points/data/*.bin
  data_poses/<sequence>/cam0_to_world.txt
```

#### Waymo Open Dataset

The loader reads the released v2 parquet directly, so no conversion step is needed; this needs
`pip install pyarrow`. Download these five perception components, keeping the distributed
layout:

```text
waymo_v2/validation/
  camera_image/<segment>.parquet
  camera_calibration/<segment>.parquet
  lidar/<segment>.parquet
  lidar_camera_projection/<segment>.parquet
  vehicle_pose/<segment>.parquet
```

The shipped clip list spans 42 `validation` segments, stratified over the split's time-of-day,
location and weather conditions (~24 GB for the five components). The segment names are the
prefixes in `eval/clips/waymo_200.txt`.

### Usage

```bash
python evaluate.py \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --dataset kitti360 --data-root /path/to/KITTI-360 \
  --output results.json
```

For Waymo, pass its root and clip list:

```bash
python evaluate.py \
  --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
  --dataset waymo --data-root /path/to/waymo_v2/validation \
  --clip-list eval/clips/waymo_200.txt \
  --output results_waymo.json
```

The clips behind the tables below are listed in `eval/clips/`; the KITTI-360 list is used by
default. Point `--clip-list` at your own file (one `<sequence>:<first frame>` per line) to score
a different set, and `--cache-dir` to reuse decoded ground truth between runs.

To reproduce every row of the tables below:

```bash
eval/run_all.sh --lfg checkpoints/lfg_seg_motion_m3n3.pt \
                --kitti360 /path/to/KITTI-360 \
                --waymo /path/to/waymo_v2/validation \
                --pi3 /path/to/pi3.safetensors
```

Waymo and Pi3 are optional — omit either flag and those rows are skipped. Individual models run
through the same harness via `--model`:

| `--model` | Predicts | Extra install |
|---|---|---|
| `lfg` (default) | depth, semantics, trajectory | none |
| `pi3` | depth, trajectory | [Pi3](https://github.com/yyfz/Pi3) on `PYTHONPATH`; pass its weights to `--checkpoint` |
| `vggt` | depth, trajectory | `pip install git+https://github.com/facebookresearch/vggt.git` |
| `da3` | depth | `pip install --no-deps git+https://github.com/ByteDance-Seed/Depth-Anything-3.git` |
| `segformer` | semantics | `pip install transformers` |
| `maskformer` | semantics | `pip install transformers` |
| `static` | semantics | none; carries the last observed frame's labels forward |

Baseline weights download automatically on first use, except Pi3, whose checkpoint you pass
explicitly. `da3` needs `--no-deps` because its declared dependencies pin an old `moviepy` and
require `xformers`, neither of which this code path uses.

### Results

200 clips per dataset at 2 Hz. Raw output, including per-metric standard deviations, is in
`eval/results/`.

#### KITTI-360

**Depth** — AbsRel, RMSE (m) and the share of pixels within a factor of 1.25, against Velodyne.

| Model | Frames seen | AbsRel | RMSE | δ<1.25 | AbsRel (pred.) | RMSE (pred.) | δ<1.25 (pred.) |
|---|---|---|---|---|---|---|---|
| Pi3 | 6 | 0.091 | 2.65 | 0.928 | 0.093 | 2.73 | 0.923 |
| VGGT | 6 | 0.100 | 2.78 | 0.919 | 0.091 | 2.87 | 0.917 |
| DA3 | 6 | 0.120 | 3.12 | 0.879 | 0.121 | 3.17 | 0.877 |
| **LFG** | 3 | 0.142 | 3.46 | 0.836 | 0.164 | 4.05 | 0.786 |

**Trajectory** — ATE after a similarity alignment; rotation and translation error against the
first frame, translation as a share of the distance travelled.

| Model | Frames seen | ATE (m) | Rot (deg) | Trans (%) |
|---|---|---|---|---|
| Pi3 | 6 | 0.09 | 0.91 | 9.5 |
| VGGT | 6 | 0.20 | 1.37 | 10.8 |
| **LFG** | 3 | 0.27 | 2.46 | 18.4 |

**Semantics** — seven classes, averaged per frame over the classes present.

| Model | Frames seen | Split | PA | mIoU |
|---|---|---|---|---|
| Static (labels carried forward) | 3 | predicted | 0.822 | 0.520 |
| MaskFormer | 6 | overall | 0.938 | 0.623 |
| SegFormer | 6 | overall | 0.952 | 0.705 |
| **LFG** | 3 | overall | 0.902 | 0.665 |
| **LFG** | 3 | predicted | 0.866 | 0.606 |

#### Waymo Open Dataset

**Depth**

| Model | Frames seen | AbsRel | RMSE | δ<1.25 | AbsRel (pred.) | RMSE (pred.) | δ<1.25 (pred.) |
|---|---|---|---|---|---|---|---|
| Pi3 | 6 | 0.140 | 5.27 | 0.842 | 0.140 | 5.26 | 0.841 |
| VGGT | 6 | 0.077 | 3.84 | 0.941 | 0.077 | 3.87 | 0.940 |
| DA3 | 6 | 0.151 | 5.57 | 0.829 | 0.152 | 5.55 | 0.829 |
| **LFG** | 3 | 0.172 | 5.95 | 0.781 | 0.184 | 6.55 | 0.764 |

**Trajectory**

| Model | Frames seen | ATE (m) | Rot (deg) | Trans (%) |
|---|---|---|---|---|
| Pi3 | 6 | 0.07 | 0.51 | 2.3 |
| VGGT | 6 | 0.06 | 0.37 | 2.0 |
| **LFG** | 3 | 0.31 | 0.81 | 13.3 |

### Conventions

- **Depth** is affine-aligned to the LiDAR once per clip, since point maps carry one unknown
  scale and shift. Ground truth beyond 80 m is ignored.
- **Semantics** uses seven classes; PA and mIoU are averaged per frame over the classes present
  in it, excluding Cityscapes void labels.
- **Trajectory** aligns predicted poses by a similarity before measuring ATE. Translation error
  is a share of distance travelled, so it does not grow with the length of the clip.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{strong2026lfg,
  title     = {Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos},
  author    = {Strong, Matthew and Chang, Wei-Jer and Herau, Quentin and Yang, Jiezhi and Hu, Yihan and Peng, Chensheng and Zhan, Wei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## ⚖️ License

- **Code:** Apache-2.0 — see [LICENSE](LICENSE).
- **Model weights:** CC BY-NC 4.0 — see the terms on [Hugging Face](https://huggingface.co/AppliedIntuitionResearch/LFG).

Third-party components under `Pi3/` are licensed separately — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 🙏 Acknowledgments

This codebase builds on [Pi3](https://github.com/yyfz/Pi3), whose model code is bundled under
`Pi3/`, and which in turn builds on [DINOv2](https://github.com/facebookresearch/dinov2)
(Meta Platforms). Evaluation baselines use [VGGT](https://github.com/facebookresearch/vggt) and
[SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024). We thank
the authors for open-sourcing their work.
