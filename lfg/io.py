from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .paths import ensure_local_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    source: str
    frame_index: int
    timestamp_sec: float | None = None


@dataclass(frozen=True)
class InputInspection:
    input: str
    kind: str
    frame_count: int | None
    sampled_frame_count: int
    first_frame: str | None = None
    last_frame: str | None = None
    first_frame_index: int | None = None
    last_frame_index: int | None = None
    fps: float | None = None
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "kind": self.kind,
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "fps": _finite_float_or_none(self.fps),
            "duration_sec": _finite_float_or_none(self.duration_sec),
        }


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _positive_finite_float_or_none(value: float) -> float | None:
    value = _finite_float_or_none(value)
    if value is None or value <= 0:
        return None
    return value


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _is_glob_pattern(value: str) -> bool:
    return any(char in value for char in "*?[]")


def _load_image(path: Path, frame_index: int) -> Frame:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return Frame(rgb=rgb, source=str(path), frame_index=frame_index)


def _load_image_sequence(paths: Iterable[Path]) -> list[Frame]:
    frames = []
    for index, path in enumerate(paths):
        frames.append(_load_image(path, index))
    return frames


def _iter_indexed_images(indexed_paths: Iterable[tuple[int, Path]]) -> Iterable[Frame]:
    for index, path in indexed_paths:
        yield _load_image(path, index)


def _list_images(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: _natural_key(path.name))


def _sampled_count(
    total_frames: int,
    *,
    frame_stride: int,
    max_frames: int | None,
    start_frame: int,
) -> int:
    stride = max(1, frame_stride)
    start = max(0, start_frame)
    if total_frames <= start:
        return 0
    count = ((total_frames - 1 - start) // stride) + 1
    if max_frames is not None:
        count = min(count, max(0, max_frames))
    return count


def _sample_indexed_paths(
    paths: list[Path],
    *,
    frame_stride: int,
    max_frames: int | None,
    start_frame: int,
) -> list[tuple[int, Path]]:
    stride = max(1, frame_stride)
    start = max(0, start_frame)
    sampled = list(enumerate(paths))[start::stride]
    if max_frames is not None:
        sampled = sampled[: max(0, max_frames)]
    return sampled


def _inspect_image_sequence(
    input_path: str,
    *,
    kind: str,
    paths: list[Path],
    frame_stride: int,
    max_frames: int | None,
    start_frame: int,
) -> InputInspection:
    sampled = _sample_indexed_paths(
        paths,
        frame_stride=frame_stride,
        max_frames=max_frames,
        start_frame=start_frame,
    )
    first = sampled[0] if sampled else None
    last = sampled[-1] if sampled else None
    return InputInspection(
        input=input_path,
        kind=kind,
        frame_count=len(paths),
        sampled_frame_count=len(sampled),
        first_frame=None if first is None else str(first[1]),
        last_frame=None if last is None else str(last[1]),
        first_frame_index=None if first is None else first[0],
        last_frame_index=None if last is None else last[0],
    )


def _inspect_video_by_scanning(
    cap: cv2.VideoCapture,
    *,
    frame_stride: int,
    max_frames: int | None,
    start_frame: int,
) -> tuple[int, int, int | None, int | None]:
    stride = max(1, frame_stride)
    start = max(0, start_frame)
    limit = None if max_frames is None else max(0, max_frames)
    absolute_index = 0
    sampled_count = 0
    first_frame_index = None
    last_frame_index = None

    while True:
        ok = cap.grab()
        if not ok:
            break
        if (
            (limit is None or sampled_count < limit)
            and absolute_index >= start
            and (absolute_index - start) % stride == 0
        ):
            if first_frame_index is None:
                first_frame_index = absolute_index
            last_frame_index = absolute_index
            sampled_count += 1
        absolute_index += 1

    return absolute_index, sampled_count, first_frame_index, last_frame_index


def _inspect_video(
    path: Path,
    *,
    frame_stride: int,
    max_frames: int | None,
    start_frame: int,
) -> InputInspection:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        fps = _positive_finite_float_or_none(cap.get(cv2.CAP_PROP_FPS))

        raw_frame_count = _positive_finite_float_or_none(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(raw_frame_count) if raw_frame_count is not None else None
        if frame_count is None:
            frame_count, sampled_count, first_index, last_index = _inspect_video_by_scanning(
                cap,
                frame_stride=frame_stride,
                max_frames=max_frames,
                start_frame=start_frame,
            )
        else:
            sampled_count = _sampled_count(
                frame_count,
                frame_stride=frame_stride,
                max_frames=max_frames,
                start_frame=start_frame,
            )
            first_index = max(0, start_frame) if sampled_count else None
            last_index = None
            if sampled_count:
                last_index = first_index + (sampled_count - 1) * max(1, frame_stride)

        return InputInspection(
            input=str(path),
            kind="video",
            frame_count=frame_count,
            sampled_frame_count=sampled_count,
            first_frame=str(path) if sampled_count else None,
            last_frame=str(path) if sampled_count else None,
            first_frame_index=first_index,
            last_frame_index=last_index,
            fps=fps,
            duration_sec=None if fps is None or frame_count is None else frame_count / fps,
        )
    finally:
        cap.release()


def inspect_input(
    input_path: str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> InputInspection:
    """Inspect local input frame counts without decoding image pixels."""

    ensure_local_path(input_path, kind="Input")
    path = Path(input_path)
    if path.exists() and path.is_dir():
        return _inspect_image_sequence(
            input_path,
            kind="image_directory",
            paths=_list_images(path),
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
    if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return _inspect_video(
            path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
    if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return _inspect_image_sequence(
            input_path,
            kind="image",
            paths=[path],
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
    if _is_glob_pattern(input_path):
        paths = [Path(item) for item in glob.glob(input_path)]
        paths = sorted(
            [item for item in paths if item.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda item: _natural_key(str(item)),
        )
        return _inspect_image_sequence(
            input_path,
            kind="image_glob",
            paths=paths,
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
    raise ValueError(
        "Input must be a video file, image file, image directory, or quoted image glob pattern."
    )


def _iter_video_frames(
    path: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Iterable[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        fps = _positive_finite_float_or_none(cap.get(cv2.CAP_PROP_FPS))

        absolute_index = 0
        kept = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if absolute_index >= start_frame and (absolute_index - start_frame) % frame_stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = None if fps is None else absolute_index / fps
                yield Frame(
                    rgb=rgb,
                    source=str(path),
                    frame_index=absolute_index,
                    timestamp_sec=timestamp,
                )
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            absolute_index += 1
    finally:
        cap.release()


def _read_video(
    path: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> list[Frame]:
    return list(
        _iter_video_frames(
            path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
    )


def iter_input_frames(
    input_path: str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Iterable[Frame]:
    """Stream sampled frames from a local video, image directory, glob, or image."""

    ensure_local_path(input_path, kind="Input")
    path = Path(input_path)
    if path.exists() and path.is_dir():
        yield from _iter_indexed_images(
            _sample_indexed_paths(
                _list_images(path),
                frame_stride=frame_stride,
                max_frames=max_frames,
                start_frame=start_frame,
            )
        )
        return
    if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        yield from _iter_video_frames(
            path,
            frame_stride=max(1, frame_stride),
            max_frames=max_frames,
            start_frame=max(0, start_frame),
        )
        return
    if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        yield from _iter_indexed_images(
            _sample_indexed_paths(
                [path],
                frame_stride=frame_stride,
                max_frames=max_frames,
                start_frame=start_frame,
            )
        )
        return
    if _is_glob_pattern(input_path):
        paths = [Path(item) for item in glob.glob(input_path)]
        paths = sorted(
            [item for item in paths if item.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda item: _natural_key(str(item)),
        )
        yield from _iter_indexed_images(
            _sample_indexed_paths(
                paths,
                frame_stride=frame_stride,
                max_frames=max_frames,
                start_frame=start_frame,
            )
        )
        return
    raise ValueError(
        "Input must be a video file, image file, image directory, or quoted image glob pattern."
    )


def load_frames(
    input_path: str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> list[Frame]:
    """Load a video, image directory, glob, or single image into RGB frames."""

    ensure_local_path(input_path, kind="Input")
    path = Path(input_path)
    if path.exists() and path.is_dir():
        frames = _load_image_sequence(_list_images(path))
    elif path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        frames = _read_video(
            path,
            frame_stride=max(1, frame_stride),
            max_frames=max_frames,
            start_frame=max(0, start_frame),
        )
        return frames
    elif path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        frames = [_load_image(path, 0)]
    elif _is_glob_pattern(input_path):
        paths = [Path(item) for item in glob.glob(input_path)]
        paths = sorted(
            [item for item in paths if item.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda item: _natural_key(str(item)),
        )
        frames = _load_image_sequence(paths)
    else:
        raise ValueError(
            "Input must be a video file, image file, image directory, or quoted image glob pattern."
        )

    if start_frame > 0:
        frames = frames[start_frame:]
    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def iter_frame_windows(
    frames: Iterable[Frame],
    *,
    history: int,
    window_stride: int,
) -> Iterable[tuple[int, list[Frame], list[bool]]]:
    if history <= 0:
        raise ValueError("history must be positive")

    stride = max(1, window_stride)
    frame_iter = iter(frames)
    cache: list[Frame] = []
    cache_start_index = 0
    target_start = 0
    exhausted = False
    last_full_window_start: int | None = None

    while True:
        required_end = target_start + history
        while not exhausted and cache_start_index + len(cache) < required_end:
            try:
                cache.append(next(frame_iter))
            except StopIteration:
                exhausted = True
                break

        total_available = cache_start_index + len(cache)
        if (
            exhausted
            and last_full_window_start is not None
            and last_full_window_start + history >= total_available
        ):
            return

        drop_count = target_start - cache_start_index
        if drop_count > 0:
            cache = cache[drop_count:]
            cache_start_index = target_start

        if not cache:
            return

        window = cache[:history]
        padded = [False] * len(window)
        if len(window) < history:
            window = window + [window[-1]] * (history - len(window))
            padded.extend([True] * (history - len(padded)))
            yield target_start, window, padded
            return

        yield target_start, window, padded
        last_full_window_start = target_start
        target_start += stride


def count_frame_windows(num_frames: int, *, history: int, window_stride: int) -> int:
    if history <= 0:
        raise ValueError("history must be positive")
    if num_frames <= 0:
        return 0
    stride = max(1, window_stride)
    count = 0
    start = 0
    while start < num_frames:
        count += 1
        if start + history >= num_frames:
            break
        start += stride
    return count


def preprocess_frames(
    frames: list[Frame],
    *,
    target_size: int = 518,
    mode: str = "crop",
    keep_ratio: bool = False,
    patch_size: int = 14,
) -> torch.Tensor:
    """Convert RGB frames to a normalized float tensor shaped [T, C, H, W]."""

    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if target_size % patch_size != 0:
        raise ValueError("target_size must be divisible by patch_size")

    tensors = []
    for frame in frames:
        image = torch.from_numpy(np.asarray(frame.rgb).copy()).permute(2, 0, 1).float() / 255.0
        tensors.append(_resize_image(image, target_size, mode, keep_ratio, patch_size))
    return torch.stack(tensors, dim=0)


def _resize_image(
    image: torch.Tensor,
    target_size: int,
    mode: str,
    keep_ratio: bool,
    patch_size: int,
) -> torch.Tensor:
    _, height, width = image.shape

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size

    new_width = max(patch_size, new_width)
    new_height = max(patch_size, new_height)

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    if mode == "pad":
        h_padding = target_size - new_height
        w_padding = target_size - new_width
        pad_top = max(0, h_padding // 2)
        pad_bottom = max(0, h_padding - pad_top)
        pad_left = max(0, w_padding // 2)
        pad_right = max(0, w_padding - pad_left)
        if pad_top or pad_bottom or pad_left or pad_right:
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
        return resized

    if not keep_ratio and new_height > target_size:
        top = (new_height - target_size) // 2
        resized = resized[:, top : top + target_size, :]
    return resized
