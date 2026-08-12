#!/usr/bin/env python3
# Copyright 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate depth, semantics and trajectory on clips of consecutive driving frames.

Each clip is six consecutive frames. LFG is given the first three and predicts all six;
baselines that are not future-predicting (pi3, vggt, segformer) are given all six, so for
them the "predicted" split simply names the same frames LFG had to extrapolate. Metrics
are reported over all frames ("overall"), the frames the model observed ("observed") and
the frames it had to predict ("predicted").

  depth       AbsRel and RMSE in metres against LiDAR, after a least-squares scale and
              shift alignment. Point maps are only defined up to one scale and shift per
              clip, so the fit is per clip by default; --alignment per-frame is available
              and moves AbsRel by under 0.01.
  semantics   pixel accuracy, mIoU, mDice and frequency-weighted IoU over seven classes.
              Averaged per frame over the classes present in that frame by default;
              --seg-average all instead averages over all seven, scoring absent classes
              zero, and --seg-metrics dataset accumulates one confusion matrix instead.
  trajectory  ATE in metres after a similarity alignment, plus rotation (deg) and
              translation (m) error of each frame's pose relative to the first.

Clips are sampled uniformly at random from every valid window in the dataset. Pass --seed
for a reproducible sample, or --clip-list to evaluate an exact, named set of clips (see
eval/clips/); --save-clip-list records whichever set was used.

Examples:
  python evaluate.py --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
      --dataset kitti360 --data-root /data/KITTI-360 --clip-list eval/clips/kitti360_200.txt

  python evaluate.py --model pi3 --checkpoint /path/to/pi3.safetensors \
      --dataset kitti360 --data-root /data/KITTI-360 --num-clips 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from tqdm import tqdm

from lfg.checkpoint import load_model_from_checkpoint
from lfg.inference import predict_window
from lfg.io import load_frame, preprocess_frames

CLIP_LENGTH = 6
IGNORE_LABEL = 255

# Whether class averages span every class or only those present in a frame.
SEG_AVERAGE = {"over_all_classes": False}

# The segmentation head's 7 classes, in index order.
CLASS_NAMES = ["road", "vehicle", "person", "traffic light", "traffic sign", "sky",
               "building/grass/background"]

# Cityscapes labelId -> class index. Cityscapes "void" ids carry no supervision and are
# excluded from the metrics rather than folded into the background class.
CITYSCAPES_LABEL_TO_CLASS = {
    7: 0,                                                    # road
    26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1,                # car/truck/bus/train/motorcycle/bicycle
    24: 2, 25: 2,                                            # person, rider
    19: 3,                                                   # traffic light
    20: 4,                                                   # traffic sign
    23: 5,                                                   # sky
    8: 6, 11: 6, 12: 6, 13: 6, 17: 6, 21: 6, 22: 6,          # sidewalk/building/wall/fence/pole/vegetation/terrain
}


@dataclass
class Clip:
    """One evaluation clip: frame images plus a lazy loader for ground-truth depth."""

    name: str
    image_paths: list[Path]
    load_depth: Callable[[int, int, int], np.ndarray]
    """load_depth(frame_slot, out_height, out_width) -> sparse metric depth, NaN where absent."""
    load_poses: Callable[[], np.ndarray] | None = None
    """load_poses() -> (CLIP_LENGTH, 4, 4) ground-truth camera-to-world poses, if available."""
    load_labels: Callable[[int, int, int], np.ndarray] | None = None
    """load_labels(frame_slot, out_height, out_width) -> class indices, IGNORE_LABEL for void."""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def affine_align(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares scale and shift taking `pred` onto `target` over `mask`."""
    source = pred[mask]
    design = np.stack([source, np.ones_like(source)], axis=1)
    scale, shift = np.linalg.lstsq(design, target[mask], rcond=None)[0]
    return scale * pred + shift


def depth_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """AbsRel and RMSE (metres) over the valid pixels of an already-aligned prediction."""
    estimate = np.clip(pred[mask], 1e-3, None)
    truth = target[mask]
    abs_rel = float(np.mean(np.abs(estimate - truth) / truth))
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    return abs_rel, rmse


def frame_segmentation_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    """Per-frame pixel accuracy, mIoU, mDice and frequency-weighted IoU.

    Class averages cover the classes present in the prediction or the ground truth of that
    frame, so a class absent from a frame neither helps nor hurts. Averaging these per-frame
    values across clips is the convention that best matches the published Table 1.
    """
    classes = len(CLASS_NAMES)
    valid = (truth != IGNORE_LABEL) & (predicted < classes)
    predicted, truth = predicted[valid], truth[valid]
    if predicted.size == 0:
        return (np.nan,) * 4

    accuracy = float((predicted == truth).mean())
    ious, dices, weights = [], [], []
    for index in range(classes):
        prediction_mask, truth_mask = predicted == index, truth == index
        union = int((prediction_mask | truth_mask).sum())
        if union == 0:
            if SEG_AVERAGE["over_all_classes"]:   # a class absent from both scores zero
                ious.append(0.0)
                dices.append(0.0)
                weights.append(0.0)
            continue
        intersection = int((prediction_mask & truth_mask).sum())
        ious.append(intersection / union)
        dices.append(2 * intersection / (int(prediction_mask.sum()) + int(truth_mask.sum())))
        weights.append(int(truth_mask.sum()) / truth.size)
    if not ious:
        return accuracy, np.nan, np.nan, np.nan
    return (accuracy, float(np.mean(ious)), float(np.mean(dices)),
            float(np.sum(np.asarray(weights) * np.asarray(ious))))


def accumulate_confusion(predicted: np.ndarray, truth: np.ndarray, into: np.ndarray) -> None:
    """Add one frame to a class-by-class confusion matrix, skipping void pixels."""
    classes = len(CLASS_NAMES)
    valid = (truth != IGNORE_LABEL) & (predicted < classes)
    predicted, truth = predicted[valid].astype(np.int64), truth[valid].astype(np.int64)
    into += np.bincount(truth * classes + predicted, minlength=classes ** 2).reshape(classes, classes)


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Pixel accuracy, mIoU, mDice and frequency-weighted IoU from a confusion matrix.

    Metrics are computed once over the accumulated matrix rather than averaged per frame,
    so a class the model rarely gets right is not hidden by the frames where it is absent.
    """
    total = confusion.sum()
    if total == 0:
        return {"pa": np.nan, "miou": np.nan, "mdice": np.nan, "fwiou": np.nan}
    true_positive = np.diag(confusion).astype(float)
    actual, predicted_total = confusion.sum(1).astype(float), confusion.sum(0).astype(float)
    union = actual + predicted_total - true_positive
    present = union > 0
    iou = np.divide(true_positive, union, out=np.zeros_like(union), where=present)
    dice = np.divide(2 * true_positive, actual + predicted_total,
                     out=np.zeros_like(union), where=(actual + predicted_total) > 0)
    return {
        "pa": float(true_positive.sum() / total),
        "miou": float(iou[present].mean()),
        "mdice": float(dice[present].mean()),
        "fwiou": float((actual[present] / total * iou[present]).sum()),
    }


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, rotation, translation) taking `source` points onto `target`."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centred, target_centred = source - source_mean, target - target_mean
    u, singular_values, vt = np.linalg.svd(target_centred.T @ source_centred / len(source))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = (source_centred ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance) if variance > 0 else 1.0
    return scale, rotation, target_mean - scale * rotation @ source_mean


def trajectory_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """ATE (m) plus relative-pose rotation (deg) and translation (m) errors for one clip.

    Both trajectories are expressed relative to their first frame. Predicted poses are only
    defined up to a similarity, so a scale is fitted against the ground-truth positions;
    ATE additionally applies the full similarity before measuring position error, while the
    relative-pose errors compare each frame's pose against frame 0.
    """
    relative = lambda poses: np.linalg.inv(poses[0]) @ poses  # noqa: E731
    predicted, truth = relative(predicted), relative(truth)
    scale, rotation, translation = umeyama(predicted[:, :3, 3], truth[:, :3, 3])

    aligned = (scale * (rotation @ predicted[:, :3, 3].T).T) + translation
    ate = float(np.sqrt((np.linalg.norm(aligned - truth[:, :3, 3], axis=1) ** 2).mean()))

    residual = np.einsum("nij,nkj->nik", predicted[1:, :3, :3], truth[1:, :3, :3])
    cosines = (np.trace(residual, axis1=1, axis2=2) - 1.0) / 2.0
    rotation_error = float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).mean())

    translation_error = float(
        np.linalg.norm(scale * predicted[1:, :3, 3] - truth[1:, :3, 3], axis=1).mean()
    )
    return ate, rotation_error, translation_error


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {"mean": float(array.mean()), "std": float(array.std()), "count": int(array.size)}


# --------------------------------------------------------------------------------------
# Dataset adapters
# --------------------------------------------------------------------------------------


def _sparse_depth_map(
    rows: np.ndarray,
    cols: np.ndarray,
    depth: np.ndarray,
    source_hw: tuple[int, int],
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Rasterise scattered depth samples onto the model's output grid, nearest surface wins."""
    source_h, source_w = source_hw
    out_h, out_w = out_hw
    y = np.clip((((rows + 0.5) / source_h) * out_h).astype(int), 0, out_h - 1)
    x = np.clip((((cols + 0.5) / source_w) * out_w).astype(int), 0, out_w - 1)
    canvas = np.full((out_h, out_w), np.nan, dtype=np.float32)
    order = np.argsort(-depth)
    canvas[y[order], x[order]] = depth[order]
    return canvas


def _kitti360_calibration(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_rect_00, T_cam0_from_velo) for the left perspective camera."""
    perspective = {}
    with (root / "calibration" / "perspective.txt").open() as handle:
        for line in handle:
            key, _, values = line.partition(":")
            if values.strip():
                perspective[key.strip()] = np.fromstring(values, sep=" ")
    projection = perspective["P_rect_00"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = perspective["R_rect_00"].reshape(3, 3)

    cam_to_velo = np.loadtxt(root / "calibration" / "calib_cam_to_velo.txt").reshape(3, 4)
    velo_from_cam = np.eye(4)
    velo_from_cam[:3, :4] = cam_to_velo
    return projection, rect @ np.linalg.inv(velo_from_cam)


def kitti360_clips(root: Path, max_depth: float) -> Iterator[Clip]:
    """KITTI-360 in its official layout, LiDAR projected into the left perspective camera.

    Layout: data_2d_raw/<drive>/image_00/data_rect/<idx>.png
            data_3d_raw/<drive>/velodyne_points/data/<idx>.bin
            calibration/{perspective.txt,calib_cam_to_velo.txt}
    """
    projection, cam_from_velo = _kitti360_calibration(root)
    for drive in sorted((root / "data_2d_raw").iterdir()):
        image_dir = drive / "image_00" / "data_rect"
        velodyne_dir = root / "data_3d_raw" / drive.name / "velodyne_points" / "data"
        if not image_dir.is_dir():
            continue
        has_lidar = velodyne_dir.is_dir()
        indices = sorted(int(p.stem) for p in image_dir.glob("*.png"))
        for start in range(len(indices) - CLIP_LENGTH + 1):
            window = indices[start : start + CLIP_LENGTH]
            if window != list(range(window[0], window[0] + CLIP_LENGTH)):
                continue
            if has_lidar and not all((velodyne_dir / f"{i:010d}.bin").exists() for i in window):
                continue

            def load_depth(
                slot: int,
                out_h: int,
                out_w: int,
                window=window,
                image_dir=image_dir,
                velodyne_dir=velodyne_dir,
                has_lidar=has_lidar,
            ) -> np.ndarray:
                from PIL import Image

                if not has_lidar:  # semantics-only run
                    return np.full((out_h, out_w), np.nan, dtype=np.float32)

                scan = np.fromfile(velodyne_dir / f"{window[slot]:010d}.bin", dtype=np.float32)
                points = scan.reshape(-1, 4)[:, :3]
                homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                in_camera = homogeneous @ cam_from_velo.T
                forward = in_camera[:, 2]
                in_camera = in_camera[forward > 0]
                pixels = in_camera @ projection.T
                depth = pixels[:, 2]
                cols, rows = pixels[:, 0] / depth, pixels[:, 1] / depth
                width, height = Image.open(image_dir / f"{window[slot]:010d}.png").size
                keep = (
                    (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height) & (depth <= max_depth)
                )
                return _sparse_depth_map(
                    rows[keep], cols[keep], depth[keep], (height, width), (out_h, out_w)
                )

            semantic_dir = root / "data_2d_semantics" / "train" / drive.name / "image_00" / "semantic"

            def load_labels(
                slot: int, out_h: int, out_w: int, window=window, semantic_dir=semantic_dir
            ) -> np.ndarray:
                from PIL import Image

                label_path = semantic_dir / f"{window[slot]:010d}.png"
                label_ids = np.asarray(
                    Image.open(label_path).resize((out_w, out_h), Image.NEAREST)
                )
                classes = np.full(label_ids.shape, IGNORE_LABEL, dtype=np.uint8)
                for label_id, class_index in CITYSCAPES_LABEL_TO_CLASS.items():
                    classes[label_ids == label_id] = class_index
                return classes

            # data_poses.zip extracts either under data_poses/ or straight to the root
            pose_file = next(
                (candidate for candidate in (
                    root / "data_poses" / drive.name / "cam0_to_world.txt",
                    root / drive.name / "cam0_to_world.txt",
                ) if candidate.exists()),
                root / "data_poses" / drive.name / "cam0_to_world.txt",
            )

            def load_poses(window=window, pose_file=pose_file) -> np.ndarray:
                table = np.loadtxt(pose_file)
                by_frame = {int(row[0]): row[1:].reshape(4, 4) for row in table}
                if not all(i in by_frame for i in window):
                    raise KeyError("clip has frames without a pose")
                return np.stack([by_frame[i] for i in window])

            if semantic_dir.is_dir() and not all(
                (semantic_dir / f"{i:010d}.png").exists() for i in window
            ):
                continue

            yield Clip(
                name=f"{drive.name}:{window[0]:010d}",
                image_paths=[image_dir / f"{i:010d}.png" for i in window],
                load_depth=load_depth,
                load_poses=load_poses if pose_file.exists() else None,
                load_labels=load_labels if semantic_dir.is_dir() else None,
            )


ADAPTERS = {"kitti360": kitti360_clips}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def load_pi3_baseline(weights: str, device: str):
    """The pi3 teacher, which is given every frame of the clip rather than only the history.

    Needs the upstream pi3 package (https://github.com/yyfz/Pi3); it is not vendored here.
    """
    try:
        from pi3.models.pi3 import Pi3
    except ImportError as exc:  # pragma: no cover - depends on an optional package
        raise SystemExit(
            "The pi3 baseline needs the upstream pi3 package on PYTHONPATH "
            "(https://github.com/yyfz/Pi3)."
        ) from exc

    model = Pi3()
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
    model.load_state_dict(state)
    return model.eval().to(device)


NEEDS_CHECKPOINT = {"lfg", "pi3"}


def build_predictor(args: argparse.Namespace):
    """Return (predict, future_start, description).

    `predict` maps a clip's image paths to model outputs. `future_start` is the first slot
    the model had to predict rather than observe: LFG only sees its history, whereas the
    pi3 baseline is given every frame, so for pi3 the split marks the same frames the paper
    reports as "predicted" even though pi3 observed them.
    """
    if args.model in NEEDS_CHECKPOINT and not args.checkpoint:
        raise SystemExit(f"--checkpoint is required for --model {args.model}.")

    if args.model == "pi3":
        model = load_pi3_baseline(args.checkpoint, args.device)

        def predict(paths: list[Path]) -> dict:
            frames = [load_frame(path, slot) for slot, path in enumerate(paths)]
            images = preprocess_frames(
                frames, target_size=args.target_size, mode=args.resize_mode, keep_ratio=False, patch_size=14
            )
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model(images.unsqueeze(0).to(args.device))

        return predict, CLIP_LENGTH // 2, {"model": "pi3", "frames_seen": CLIP_LENGTH}

    if args.model == "segformer":
        processor, model = load_segformer_baseline(args.device)

        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        def predict(paths: list[Path]) -> dict:
            # Fed at the same input resolution as LFG so the comparison is like for like,
            # rather than at the processor's much larger default.
            frames = [load_frame(path, slot) for slot, path in enumerate(paths)]
            images = preprocess_frames(
                frames, target_size=args.target_size, mode=args.resize_mode,
                keep_ratio=False, patch_size=14,
            )
            pixels = ((images - mean) / std).to(args.device)
            with torch.inference_mode():
                logits = model(pixel_values=pixels).logits      # (S, 19, h, w)
            classes = torch.full(
                (logits.shape[0], len(CLASS_NAMES), *logits.shape[-2:]), -1e4, device=logits.device
            )
            for train_id, index in CITYSCAPES_TRAIN_ID_TO_CLASS.items():
                classes[:, index] = torch.maximum(classes[:, index], logits[:, train_id])
            return {"segmentation": classes.permute(0, 2, 3, 1).unsqueeze(0)}

        return predict, CLIP_LENGTH // 2, {"model": "segformer", "frames_seen": CLIP_LENGTH}

    if args.model == "static":
        # No network: the observed frames' labels are carried forward unchanged, which is
        # the paper's "static" reference for how much of the future is simply the present.
        def predict(paths: list[Path]) -> dict:
            return {"static": True}

        return predict, CLIP_LENGTH // 2, {"model": "static", "frames_seen": CLIP_LENGTH // 2}

    if args.model == "vggt":
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = load_vggt_baseline(args.device)

        def predict(paths: list[Path]) -> dict:
            frames = [load_frame(path, slot) for slot, path in enumerate(paths)]
            images = preprocess_frames(
                frames, target_size=args.target_size, mode=args.resize_mode, keep_ratio=False, patch_size=14
            ).unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
            depth = outputs["depth"][..., 0]                        # (1, S, H, W)
            points = torch.zeros(*depth.shape, 3, device=depth.device)
            points[..., 2] = depth
            extrinsic, _ = pose_encoding_to_extri_intri(outputs["pose_enc"], images.shape[-2:])
            square = torch.eye(4, device=depth.device).repeat(1, extrinsic.shape[1], 1, 1)
            square[:, :, :3, :4] = extrinsic                        # world-to-camera
            return {"local_points": points, "camera_poses": torch.linalg.inv(square)}

        return predict, CLIP_LENGTH // 2, {"model": "vggt", "frames_seen": CLIP_LENGTH}

    model, config, _, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)
    if config.m + config.n < CLIP_LENGTH:
        print(
            f"Warning: checkpoint predicts {config.m}+{config.n} frames; "
            f"only those are scored against the {CLIP_LENGTH}-frame clips."
        )

    def predict(paths: list[Path]) -> dict:
        frames = [load_frame(path, slot) for slot, path in enumerate(paths[: config.m])]
        return predict_window(
            model, frames, config, device=args.device,
            target_size=args.target_size, resize_mode=args.resize_mode, keep_ratio=False,
        )

    return predict, config.m, {"model": "lfg", **config.to_dict()}


# SegFormer is trained on Cityscapes train ids; this is the same grouping the LFG
# segmentation head was distilled with.
CITYSCAPES_TRAIN_ID_TO_CLASS = {
    0: 0,                                              # road
    6: 3, 7: 4, 10: 5,                                 # traffic light, traffic sign, sky
    11: 2, 12: 2,                                      # person, rider
    13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,          # car/truck/bus/train/motorcycle/bicycle
    1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 8: 6, 9: 6,          # sidewalk/building/wall/fence/pole/vegetation/terrain
}


def load_segformer_baseline(device: str):
    """The SegFormer teacher, given every RGB frame of the clip."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    return (SegformerImageProcessor.from_pretrained(name),
            SegformerForSemanticSegmentation.from_pretrained(name).eval().to(device))


def load_vggt_baseline(device: str):
    """VGGT, a depth/pose baseline that -- like pi3 -- is given every frame of the clip."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:  # pragma: no cover - depends on an optional package
        raise SystemExit(
            "The vggt baseline needs the vggt package "
            "(pip install git+https://github.com/facebookresearch/vggt.git)."
        ) from exc
    return VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)


def evaluate(args: argparse.Namespace) -> dict:
    SEG_AVERAGE["over_all_classes"] = args.seg_average == "all"
    predict, future_start, model_info = build_predictor(args)

    root = Path(args.data_root)
    missing = [name for name in ("data_2d_raw", "calibration") if not (root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"{root} does not look like a KITTI-360 root: missing {', '.join(missing)}. "
            "See the Evaluation section of the README for the expected layout."
        )
    clips = list(ADAPTERS[args.dataset](root, args.max_depth))
    if not clips:
        raise SystemExit(f"No usable {CLIP_LENGTH}-frame clips found under {args.data_root}")
    if args.clip_list:
        wanted = [line.strip() for line in Path(args.clip_list).read_text().splitlines() if line.strip()]
        by_name = {clip.name: clip for clip in clips}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(f"{len(missing)} clip(s) from {args.clip_list} not found, e.g. {missing[0]}")
        clips = [by_name[name] for name in wanted]
        print(f"{len(clips)} clips from {args.clip_list} | {args.alignment} alignment")
    else:
        random.Random(args.seed).shuffle(clips)
        clips = clips[: args.num_clips]
        print(f"{len(clips)} clips | {args.alignment} alignment | seed {args.seed}")
    if args.save_clip_list:
        Path(args.save_clip_list).write_text("\n".join(clip.name for clip in clips) + "\n")
        print(f"Wrote clip list to {args.save_clip_list}")

    classes = len(CLASS_NAMES)
    confusion = {"overall": np.zeros((classes, classes), np.int64),
                 "predicted": np.zeros((classes, classes), np.int64)}
    scored: dict[str, list[float]] = {k: [] for k in ("overall_absrel", "overall_rmse",
                                                      "predicted_absrel", "predicted_rmse",
                                                      "observed_absrel", "observed_rmse",
                                                      "trajectory_ate", "trajectory_rot",
                                                      "trajectory_trans")
                                     + tuple(f"{split}_{metric}"
                                             for split in ("overall", "predicted")
                                             for metric in ("pa", "miou", "mdice", "fwiou"))}
    for clip in tqdm(clips, desc="evaluating"):
        outputs = predict(clip.image_paths)
        if "local_points" in outputs:
            predicted = outputs["local_points"][0].float().cpu().numpy()[..., 2]
            slots = min(CLIP_LENGTH, predicted.shape[0])
            height, width = predicted.shape[1], predicted.shape[2]
        else:  # segmentation-only baselines: score on the same grid the models use
            from PIL import Image

            predicted = None
            slots = CLIP_LENGTH
            source_w, source_h = Image.open(clip.image_paths[0]).size
            width = args.target_size
            height = max(14, round(source_h * (width / source_w) / 14) * 14)

        if clip.load_poses is not None and "camera_poses" in outputs:
            try:
                gt_poses = clip.load_poses()[:slots]
                pred_poses = outputs["camera_poses"][0].float().cpu().numpy()[:slots]
                ate, rot, trans = trajectory_errors(pred_poses, gt_poses)
                scored["trajectory_ate"].append(ate)
                scored["trajectory_rot"].append(rot)
                scored["trajectory_trans"].append(trans)
            except (KeyError, OSError, ValueError):
                pass

        if clip.load_labels is not None and ("segmentation" in outputs or outputs.get("static")):
            logits = None
            if not outputs.get("static"):
                raw = outputs["segmentation"][0].permute(0, 3, 1, 2).float()   # (S, C, h, w)
                if raw.shape[-2:] != (height, width):
                    raw = torch.nn.functional.interpolate(
                        raw, size=(height, width), mode="bilinear", align_corners=False
                    )
                logits = raw.permute(0, 2, 3, 1).cpu().numpy()[:slots]
            per_frame_seg = []
            for slot in range(slots):
                try:
                    labels = clip.load_labels(slot, height, width)
                except (OSError, ValueError):
                    continue
                if logits is None:  # static: carry the last observed frame's labels forward
                    prediction = clip.load_labels(future_start - 1, height, width)
                else:
                    prediction = logits[slot].argmax(-1)
                accumulate_confusion(prediction, labels, confusion["overall"])
                per_frame_seg.append(frame_segmentation_metrics(prediction, labels))
                if slot >= future_start:
                    accumulate_confusion(prediction, labels, confusion["predicted"])

            seg = np.array(per_frame_seg, dtype=float) if per_frame_seg else np.zeros((0, 4))
            for split, rows in (("overall", seg), ("predicted", seg[future_start:])):
                if len(rows) and np.isfinite(rows[:, 0]).any():
                    for column, metric in enumerate(("pa", "miou", "mdice", "fwiou")):
                        scored[f"{split}_{metric}"].append(np.nanmean(rows[:, column]))

        if predicted is None:
            continue
        truth = [clip.load_depth(slot, height, width) for slot in range(slots)]
        masks = [np.isfinite(gt) & (np.nan_to_num(gt) > 0) for gt in truth]

        if args.alignment == "per-clip":
            stacked_mask = np.stack(masks)
            if stacked_mask.sum() < args.min_points:
                continue
            aligned = list(
                affine_align(predicted[:slots], np.nan_to_num(np.stack(truth)), stacked_mask)
            )
        else:
            aligned = [
                affine_align(predicted[slot], np.nan_to_num(truth[slot]), masks[slot])
                if masks[slot].sum() >= args.min_points
                else np.full_like(predicted[slot], np.nan)
                for slot in range(slots)
            ]

        per_frame = []
        for slot in range(slots):
            if masks[slot].sum() < args.min_points or not np.isfinite(aligned[slot]).any():
                per_frame.append((np.nan, np.nan))
                continue
            per_frame.append(depth_errors(aligned[slot], truth[slot], masks[slot]))

        errors = np.array(per_frame, dtype=float)
        if not np.isfinite(errors[:, 0]).any():
            continue
        observed, future = errors[:future_start], errors[future_start:]
        scored["overall_absrel"].append(np.nanmean(errors[:, 0]))
        scored["overall_rmse"].append(np.nanmean(errors[:, 1]))
        if len(observed) and np.isfinite(observed[:, 0]).any():
            scored["observed_absrel"].append(np.nanmean(observed[:, 0]))
            scored["observed_rmse"].append(np.nanmean(observed[:, 1]))
        if len(future) and np.isfinite(future[:, 0]).any():
            scored["predicted_absrel"].append(np.nanmean(future[:, 0]))
            scored["predicted_rmse"].append(np.nanmean(future[:, 1]))

    results = {name: summarize(values) for name, values in scored.items()}
    if args.seg_metrics == "dataset":
        for split, matrix in confusion.items():
            for metric, value in confusion_metrics(matrix).items():
                results[f"{split}_{metric}"] = {"mean": value, "std": 0.0,
                                                "count": int(matrix.sum())}
    return {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "clips_scored": results["overall_absrel"]["count"],
        "clips_requested": args.num_clips,
        "seed": None if args.clip_list else args.seed,
        "clip_list": args.clip_list,
        "alignment": args.alignment,
        "max_depth": args.max_depth,
        "model": model_info,
        "metrics": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="Path to the model weights.")
    parser.add_argument(
        "--model",
        choices=["lfg", "pi3", "vggt", "segformer", "static"],
        default="lfg",
        help="lfg (default) sees only the history; the pi3 baseline is given every frame.",
    )
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS), help="Dataset adapter.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--num-clips", type=int, default=200, help="Clips to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for clip sampling.")
    parser.add_argument(
        "--clip-list",
        default=None,
        help="Evaluate exactly these clips (one name per line) instead of sampling; makes a "
             "run reproducible across machines and comparable across models.",
    )
    parser.add_argument("--save-clip-list", default=None, help="Write the evaluated clip names here.")
    parser.add_argument(
        "--alignment",
        choices=["per-clip", "per-frame"],
        default="per-clip",
        help="Fit the scale and shift once per clip (default) or independently per frame.",
    )
    parser.add_argument(
        "--seg-average", choices=["present", "all"], default="present",
        help="Average IoU/Dice over the classes present in a frame (default), or over all "
             "seven with absent classes scoring zero.",
    )
    parser.add_argument(
        "--seg-metrics",
        choices=["per-frame", "dataset"],
        default="per-frame",
        help="Average segmentation metrics per frame (default) or compute them once over a "
             "confusion matrix accumulated across the whole run.",
    )
    parser.add_argument("--max-depth", type=float, default=80.0, help="Ignore ground truth beyond this range.")
    parser.add_argument("--min-points", type=int, default=100, help="Minimum valid pixels to score a frame.")
    parser.add_argument(
        "--resize-mode", choices=["crop", "pad"], default="crop",
        help="How frames are fitted to the model input: preserve aspect and crop, or pad.",
    )
    parser.add_argument("--target-size", type=int, default=518, help="Inference width; must be divisible by 14.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Write results JSON here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    metrics = results["metrics"]
    print(f"\n{results['clips_scored']} clips scored ({results['dataset']})")
    print(f"{'split':<12}{'AbsRel':>18}{'RMSE (m)':>18}")
    for split in ("overall", "observed", "predicted"):
        absrel, rmse = metrics[f"{split}_absrel"], metrics[f"{split}_rmse"]
        print(f"{split:<12}{absrel['mean']:>10.3f} ±{absrel['std']:<6.3f}{rmse['mean']:>10.2f} ±{rmse['std']:<6.2f}")

    if metrics["overall_pa"]["count"]:
        print(f"\n{'segmentation':<12}{'PA':>10}{'mIoU':>10}{'mDice':>10}{'FW-IoU':>10}")
        for split in ("overall", "predicted"):
            row = "".join(f"{metrics[f'{split}_{m}']['mean']:>10.3f}"
                          for m in ("pa", "miou", "mdice", "fwiou"))
            print(f"{split:<12}{row}")

    if metrics["trajectory_ate"]["count"]:
        ate, rot, trans = (metrics[f"trajectory_{k}"] for k in ("ate", "rot", "trans"))
        print(f"\ntrajectory ({ate['count']} clips){'ATE (m)':>13}{'Rot (deg)':>18}{'Trans (m)':>18}")
        print(f"{'':<24}{ate['mean']:>7.2f} ±{ate['std']:<5.2f}{rot['mean']:>10.2f} ±{rot['std']:<6.2f}"
              f"{trans['mean']:>10.2f} ±{trans['std']:<6.2f}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
