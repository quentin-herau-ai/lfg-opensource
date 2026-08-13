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
inference path and the evaluation harness used for the paper's KITTI-360 benchmarks.

## 🔥 News

- **[2026-08-12]** — Evaluation code released.
- **[2026-07-13]** — Checkpoint released on [Hugging Face](https://huggingface.co/AppliedIntuitionResearch/LFG).
- **[2026-06-14]** — Inference code released.
- **[2026-02-25]** — Paper on [arXiv](https://arxiv.org/abs/2602.22091); accepted at CVPR 2026.

## 📋 Table of Contents

- [Installation](#️-installation)
- [Checkpoints](#-checkpoints)
- [Getting Started](#-getting-started)
- [Evaluation](#-evaluation)
- [Results](#-results)
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

**External assets:** the checkpoint (below, from Hugging Face) and, for evaluation only, the
KITTI-360 dataset (see [Evaluation](#-evaluation)). Baseline comparisons additionally need their
upstream packages, listed in that section.

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

For long videos or image sequences, inference streams sampled frames through sliding windows instead of decoding the full input into memory first. The first `M` predictions correspond to the input/history frames for that window; the next `N` predictions are autoregressive future predictions. The JSON metadata records the source frame indices and which slots are padded for short tail windows.

## 📊 Evaluation

`evaluate.py` scores depth, semantic segmentation and trajectory on KITTI-360 and the Waymo
Open Dataset, following the protocol described in the paper. Each clip is six consecutive frames; LFG is given the first three and predicts
all six, so results are reported over all frames (`overall`) and over the three it had to
predict (`predicted`). Baselines that do not predict the future are given all six frames.

### Data

Two datasets are supported: [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) for depth,
segmentation and trajectory, and the [Waymo Open Dataset](https://waymo.com/open/) for depth and
trajectory. Both require registration on their respective sites.

#### KITTI-360

Follow the download
instructions on the official site to obtain the perspective images, Velodyne scans,
calibrations, vehicle poses and 2D semantic labels, and unpack them into a single dataset
root. The clip list shipped with this repo covers sequences `2013_05_28_drive_0000_sync` and
`2013_05_28_drive_0002_sync` (~50 GB). The sequences a clip list needs are always its name
prefixes, so `cut -d: -f1 eval/clips/kitti360_200.txt | sort -u` lists them.

```text
KITTI-360/
  calibration/
  data_2d_raw/<sequence>/image_00/data_rect/*.png
  data_2d_semantics/train/<sequence>/image_00/semantic/*.png
  data_3d_raw/<sequence>/velodyne_points/data/*.bin
  data_poses/<sequence>/cam0_to_world.txt
```

#### Waymo Open Dataset

The loader reads the released v2 parquet directly, so no conversion step is needed. Download
these five perception components, keeping the distributed layout:

```text
waymo_v2/validation/
  camera_image/<segment>.parquet
  camera_calibration/<segment>.parquet
  lidar/<segment>.parquet
  lidar_camera_projection/<segment>.parquet
  vehicle_pose/<segment>.parquet
```

The shipped clip list spans 42 `validation` segments, stratified over the split's time-of-day,
location and weather conditions (~24 GB for the five components). List exactly which ones you
need with:

```bash
cut -d: -f1 eval/clips/waymo_200.txt | sort -u
```

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
a different set. Baselines run through the same harness via `--model`:

| `--model` | Requires |
|---|---|
| `lfg` (default) | this repo |
| `pi3` | [Pi3](https://github.com/yyfz/Pi3) on `PYTHONPATH`, `--checkpoint` its weights |
| `vggt` | `pip install git+https://github.com/facebookresearch/vggt.git` |
| `segformer` | `transformers` |
| `maskformer` | `transformers` |
| `static` | nothing; carries the last observed frame's labels forward |

### Results

200 clips per dataset at 2 Hz (six frames spanning 2.5 s), per-clip scale-and-shift alignment,
518-wide inputs. Raw output in `eval/results/`.

#### KITTI-360

**Depth** (AbsRel, RMSE in metres, against Velodyne)

| Model | Frames seen | AbsRel | RMSE | AbsRel (pred.) | RMSE (pred.) |
|---|---|---|---|---|---|
| Pi3 | 6 | 0.091 ± 0.039 | 2.65 ± 0.96 | 0.093 ± 0.045 | 2.73 ± 1.08 |
| VGGT | 6 | 0.100 ± 0.033 | 2.78 ± 0.91 | 0.091 ± 0.033 | 2.87 ± 1.04 |
| **LFG** | 3 | 0.142 ± 0.051 | 3.46 ± 1.11 | 0.164 ± 0.061 | 4.05 ± 1.45 |

**Trajectory** (ATE after similarity alignment; rotation and translation error relative to the
first frame)

| Model | Frames seen | ATE (m) | Rot (deg) | Trans (m) |
|---|---|---|---|---|
| Pi3 | 6 | 0.09 ± 0.05 | 0.91 ± 0.97 | 1.20 ± 0.37 |
| VGGT | 6 | 0.20 ± 0.11 | 1.37 ± 1.56 | 1.35 ± 0.38 |
| **LFG** | 3 | 0.27 ± 0.14 | 2.46 ± 2.64 | 2.30 ± 0.67 |

**Semantics** (seven classes, averaged per frame over the classes present)

| Model | Frames seen | Split | PA | mIoU | mDice | FW-IoU |
|---|---|---|---|---|---|---|
| Static (labels carried forward) | – | predicted | 0.822 | 0.520 | 0.607 | 0.732 |
| MaskFormer | 6 | overall | 0.938 | 0.623 | 0.678 | 0.892 |
| SegFormer | 6 | overall | 0.952 | 0.705 | 0.760 | 0.913 |
| **LFG** | 3 | overall | 0.902 | 0.665 | 0.741 | 0.841 |
| **LFG** | 3 | predicted | 0.866 | 0.606 | 0.701 | 0.787 |

#### Waymo Open Dataset

**Depth** (AbsRel, RMSE in metres, against LiDAR)

| Model | Frames seen | AbsRel | RMSE | AbsRel (pred.) | RMSE (pred.) |
|---|---|---|---|---|---|
| Pi3 | 6 | 0.145 ± 0.094 | 5.32 ± 2.00 | 0.146 ± 0.096 | 5.31 ± 2.05 |
| VGGT | 6 | 0.080 ± 0.042 | 3.89 ± 1.12 | 0.080 ± 0.043 | 3.92 ± 1.20 |
| **LFG** | 3 | 0.172 ± 0.087 | 5.85 ± 1.87 | 0.184 ± 0.095 | 6.37 ± 2.11 |

**Trajectory**

| Model | Frames seen | ATE (m) | Rot (deg) | Trans (m) |
|---|---|---|---|---|
| Pi3 | 6 | 0.07 ± 0.12 | 0.48 ± 0.91 | 0.35 ± 0.40 |
| VGGT | 6 | 0.05 ± 0.08 | 0.34 ± 0.48 | 0.32 ± 0.35 |
| **LFG** | 3 | 0.26 ± 0.16 | 0.73 ± 1.11 | 0.60 ± 0.48 |

Seeing only three frames, LFG stays close to Pi3 given all six — within 0.8 m RMSE on KITTI-360
and 0.5 m on Waymo — while its predicted frames degrade only modestly against its observed ones.

### Conventions

Depth is aligned to ground truth with a least-squares scale and shift. Point maps are
predicted up to a single unknown scale and shift per clip, so the fit is made once per clip;
`--alignment per-frame` fits each frame separately and shifts AbsRel by under 0.01. Clips are
sampled at 2 Hz (`--frame-stride 5` on these 10 Hz datasets), one of the rates the released
checkpoint was trained on and the rate at which its encoder is used downstream; a shorter
horizon makes future prediction close to trivial. Segmentation class averages cover the classes
present in each frame; `--seg-average all` averages over all seven with absent
classes scoring zero, and `--seg-metrics dataset` accumulates one confusion matrix instead.
Ground truth beyond `--max-depth` (80 m) is ignored.

## 📈 Results

See [Evaluation](#-evaluation) for the full tables. Headline: given only three observed frames,
LFG reaches 0.106 AbsRel / 2.98 m RMSE on KITTI-360 depth — within ~0.2 m RMSE of Pi3 given all
six — and scores its three predicted frames as well as the ones it observed.

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

**Support:** best-effort via GitHub Issues. First response within ~1 week for critical bugs
blocking installation, inference, or smoke tests.
