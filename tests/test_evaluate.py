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

"""Unit tests for the evaluation metrics.

Run with `pytest tests/`. These need no dataset and no model weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (  # noqa: E402
    CLASS_NAMES,
    IGNORE_LABEL,
    affine_align,
    depth_errors,
    frame_segmentation_metrics,
    trajectory_errors,
    umeyama,
)


# --------------------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------------------


def test_affine_align_recovers_a_known_scale_and_shift():
    truth = np.linspace(2.0, 60.0, 400).reshape(20, 20)
    scrambled = (truth - 7.0) / 3.0
    aligned = affine_align(scrambled, truth, np.ones_like(truth, bool))
    assert np.allclose(aligned, truth, atol=1e-6)


def test_depth_errors_are_zero_for_a_perfect_prediction():
    truth = np.linspace(2.0, 60.0, 400).reshape(20, 20)
    abs_rel, rmse = depth_errors(truth.copy(), truth, np.ones_like(truth, bool))
    assert abs_rel == pytest.approx(0.0)
    assert rmse == pytest.approx(0.0)


def test_depth_errors_match_a_hand_computed_case():
    truth = np.array([[10.0, 20.0]])
    predicted = np.array([[11.0, 18.0]])
    mask = np.ones_like(truth, bool)
    abs_rel, rmse = depth_errors(predicted, truth, mask)
    assert abs_rel == pytest.approx((0.1 + 0.1) / 2)
    assert rmse == pytest.approx(np.sqrt((1 + 4) / 2))


def test_alignment_only_uses_masked_pixels():
    truth = np.array([[10.0, 20.0, 1e6]])          # third pixel is nonsense
    predicted = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, True, False]])
    aligned = affine_align(predicted, truth, mask)
    assert aligned[0, :2] == pytest.approx([10.0, 20.0])


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def test_segmentation_metrics_are_one_for_a_perfect_prediction():
    truth = np.array([[0, 0, 1], [1, 5, 5], [6, 6, 2]], np.uint8)
    accuracy, miou, mdice, fwiou = frame_segmentation_metrics(truth.copy(), truth)
    assert (accuracy, miou, mdice, fwiou) == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_segmentation_metrics_are_zero_when_every_pixel_is_wrong():
    truth = np.zeros((4, 4), np.uint8)
    predicted = np.full((4, 4), 3, np.uint8)
    accuracy, miou, mdice, _ = frame_segmentation_metrics(predicted, truth)
    assert (accuracy, miou, mdice) == pytest.approx((0.0, 0.0, 0.0))


def test_void_pixels_are_excluded():
    truth = np.array([[0, 0], [0, IGNORE_LABEL]], np.uint8)
    predicted = np.array([[0, 0], [0, 3]], np.uint8)   # only the void pixel disagrees
    accuracy, miou, _, _ = frame_segmentation_metrics(predicted, truth)
    assert accuracy == pytest.approx(1.0)
    assert miou == pytest.approx(1.0)


def test_pixel_accuracy_counts_agreeing_pixels():
    truth = np.array([[0, 0, 0, 1]], np.uint8)
    predicted = np.array([[0, 0, 0, 6]], np.uint8)
    accuracy, *_ = frame_segmentation_metrics(predicted, truth)
    assert accuracy == pytest.approx(0.75)


def test_class_list_matches_the_segmentation_head():
    assert len(CLASS_NAMES) == 7


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------


def _straight_line(step: float = 1.0) -> np.ndarray:
    poses = np.stack([np.eye(4) for _ in range(6)])
    poses[:, 2, 3] = np.arange(6) * step         # drive forward along z
    return poses


def test_trajectory_errors_vanish_for_an_exact_match():
    truth = _straight_line()
    ate, rotation, translation = trajectory_errors(truth.copy(), truth)
    assert (ate, rotation, translation) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_trajectory_is_scored_up_to_scale():
    """Predictions are only defined up to a similarity, so a scaled trajectory is exact."""
    truth = _straight_line(1.0)
    ate, _, translation = trajectory_errors(_straight_line(0.1), truth)
    assert ate == pytest.approx(0.0, abs=1e-6)
    assert translation == pytest.approx(0.0, abs=1e-6)


def test_rotation_error_is_reported_in_degrees():
    truth = _straight_line()
    predicted = truth.copy()
    angle = np.deg2rad(10.0)
    turn = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
    predicted[3:, :3, :3] = turn                 # three of the five scored frames are turned
    _, rotation, _ = trajectory_errors(predicted, truth)
    assert rotation == pytest.approx(10.0 * 3 / 5, abs=1e-3)


def test_umeyama_recovers_a_known_similarity():
    source = np.random.default_rng(0).normal(size=(8, 3))
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    target = 2.5 * (rotation @ source.T).T + np.array([1.0, -2.0, 3.0])
    scale, recovered, translation = umeyama(source, target)
    assert scale == pytest.approx(2.5)
    assert np.allclose(recovered, rotation, atol=1e-6)
    assert np.allclose(translation, [1.0, -2.0, 3.0], atol=1e-6)
