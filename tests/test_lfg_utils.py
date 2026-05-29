import json
import tempfile
import unittest
import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from lfg.checkpoint import inspect_checkpoint, load_checkpoint_file, load_model_from_checkpoint
from lfg.cli import prepare_output_dir, resolve_device, validate_args, validate_output_path
from lfg.config import (
    checkpoint_has_autoregressive_state,
    checkpoint_looks_multiview,
    model_config_from_checkpoint,
)
from lfg.io import (
    Frame,
    InputInspection,
    count_frame_windows,
    inspect_input,
    iter_frame_windows,
    iter_input_frames,
    load_frames,
    preprocess_frames,
)
from lfg.inference import save_window_result, window_metadata
from Pi3.pi3.models.dinov2.hub.backbones import _make_dinov2_model


class UnsupportedPickledPayload:
    pass


class InputLoadingTests(unittest.TestCase):
    def test_uri_style_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local filesystem path"):
            load_frames("abc://example/frames")

    def test_image_directory_uses_natural_order_and_pads_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in [10, 1, 2]:
                image = np.full((24, 32, 3), index, dtype=np.uint8)
                Image.fromarray(image).save(root / f"frame_{index}.png")

            frames = load_frames(str(root))
            self.assertEqual([Path(frame.source).name for frame in frames], ["frame_1.png", "frame_2.png", "frame_10.png"])

            windows = list(iter_frame_windows(frames, history=4, window_stride=4))
            self.assertEqual(len(windows), 1)
            _, window_frames, padded = windows[0]
            self.assertEqual([frame.frame_index for frame in window_frames], [0, 1, 2, 2])
            self.assertEqual(padded, [False, False, False, True])

            tensor = preprocess_frames(window_frames, target_size=42, patch_size=14)
            self.assertEqual(tuple(tensor.shape), (4, 3, 28, 42))

            with self.assertRaisesRegex(ValueError, "divisible by patch_size"):
                preprocess_frames(window_frames, target_size=40, patch_size=14)

    def test_glob_input_uses_natural_order_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in [10, 1, 2, 3]:
                image = np.full((16, 16, 3), index, dtype=np.uint8)
                Image.fromarray(image).save(root / f"frame_{index}.jpg")

            frames = load_frames(str(root / "*.jpg"), start_frame=1, frame_stride=2, max_frames=2)
            self.assertEqual([Path(frame.source).name for frame in frames], ["frame_2.jpg", "frame_10.jpg"])
            self.assertEqual([frame.frame_index for frame in frames], [1, 3])

            streamed = list(iter_input_frames(str(root / "*.jpg"), start_frame=1, frame_stride=2, max_frames=2))
            self.assertEqual([Path(frame.source).name for frame in streamed], ["frame_2.jpg", "frame_10.jpg"])
            self.assertEqual([frame.frame_index for frame in streamed], [1, 3])

            inspection = inspect_input(str(root / "*.jpg"), start_frame=1, frame_stride=2, max_frames=2)
            self.assertEqual(inspection.kind, "image_glob")
            self.assertEqual(inspection.frame_count, 4)
            self.assertEqual(inspection.sampled_frame_count, 2)
            self.assertEqual(Path(inspection.first_frame).name, "frame_2.jpg")
            self.assertEqual(Path(inspection.last_frame).name, "frame_10.jpg")
            self.assertEqual(inspection.first_frame_index, 1)
            self.assertEqual(inspection.last_frame_index, 3)

    def test_video_input_respects_start_stride_and_max_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16))
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is unavailable")
            for index in range(6):
                frame = np.full((16, 16, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            frames = load_frames(str(video_path), start_frame=1, frame_stride=2, max_frames=2)
            self.assertEqual([frame.frame_index for frame in frames], [1, 3])
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0].timestamp_sec, 0.2)
            self.assertAlmostEqual(frames[1].timestamp_sec, 0.6)

            inspection = inspect_input(str(video_path), start_frame=1, frame_stride=2, max_frames=2)
            self.assertEqual(inspection.kind, "video")
            self.assertEqual(inspection.frame_count, 6)
            self.assertEqual(inspection.sampled_frame_count, 2)
            self.assertEqual(inspection.first_frame_index, 1)
            self.assertEqual(inspection.last_frame_index, 3)
            self.assertAlmostEqual(inspection.fps, 5.0)
            self.assertAlmostEqual(inspection.duration_sec, 1.2)

    def test_input_inspection_metadata_is_strict_json_safe(self):
        inspection = InputInspection(
            input="clip.mp4",
            kind="video",
            frame_count=10,
            sampled_frame_count=10,
            fps=float("nan"),
            duration_sec=float("inf"),
        )

        payload = inspection.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["fps"])
        self.assertIsNone(payload["duration_sec"])

    def test_streaming_window_iteration_matches_list_semantics(self):
        cases = [
            (0, 4, 4, [], []),
            (3, 4, 4, [0], [[False, False, False, True]]),
            (4, 4, 4, [0], [[False, False, False, False]]),
            (5, 4, 1, [0, 1], [[False, False, False, False], [False, False, False, False]]),
            (6, 4, 4, [0, 4], [[False, False, False, False], [False, False, True, True]]),
            (10, 4, 4, [0, 4, 8], [[False, False, False, False], [False, False, False, False], [False, False, True, True]]),
            (10, 4, 5, [0, 5], [[False, False, False, False], [False, False, False, False]]),
        ]
        for length, history, stride, expected_starts, expected_padded in cases:
            with self.subTest(length=length, history=history, stride=stride):
                frames = [
                    Frame(
                        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                        source=f"frame_{index}",
                        frame_index=index,
                    )
                    for index in range(length)
                ]
                windows = list(iter_frame_windows(iter(frames), history=history, window_stride=stride))
                self.assertEqual([start for start, _, _ in windows], expected_starts)
                self.assertEqual([padded for _, _, padded in windows], expected_padded)
                self.assertEqual(count_frame_windows(length, history=history, window_stride=stride), len(windows))


class BundledModelTests(unittest.TestCase):
    def test_dinov2_builder_rejects_pretrained_downloads(self):
        with self.assertRaisesRegex(ValueError, "offline"):
            _make_dinov2_model(pretrained=True)


class DeviceTests(unittest.TestCase):
    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid device"):
            resolve_device("not-a-device")

    def test_unsupported_device_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            resolve_device("mps")

    def test_cuda_device_is_rejected_when_unavailable(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            resolve_device("cuda")

    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_device_index_must_exist(self, _is_available, _device_count):
        self.assertEqual(resolve_device("cuda:0"), "cuda:0")
        with self.assertRaisesRegex(ValueError, "only 1 CUDA device"):
            resolve_device("cuda:1")


class CliValidationTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "frame_stride": 1,
            "start_frame": 0,
            "max_frames": None,
            "window_stride": None,
            "target_size": 518,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_numeric_cli_options_are_not_silently_clamped(self):
        invalid_cases = [
            ({"frame_stride": 0}, "--frame-stride"),
            ({"start_frame": -1}, "--start-frame"),
            ({"max_frames": 0}, "--max-frames"),
            ({"window_stride": 0}, "--window-stride"),
            ({"target_size": 0}, "--target-size"),
            ({"target_size": 500}, "--target-size"),
        ]
        for overrides, option_name in invalid_cases:
            with self.subTest(option_name=option_name):
                with self.assertRaisesRegex(ValueError, option_name):
                    validate_args(self._args(**overrides))

    def test_output_path_must_be_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out"
            output_path.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                prepare_output_dir(output_path)
            with self.assertRaisesRegex(ValueError, "not a directory"):
                validate_output_path(output_path)


class OutputSavingTests(unittest.TestCase):
    def test_window_metadata_is_strict_json_safe(self):
        frame = Frame(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            source="frame_000.png",
            frame_index=0,
            timestamp_sec=float("nan"),
        )

        metadata = window_metadata(
            window_index=0,
            start_index=0,
            frames=[frame],
            padded=[False],
            config=model_config_from_checkpoint({"MODEL": {"M": 1, "N": 0}}, {}),
        )

        json.dumps(metadata, allow_nan=False)
        self.assertIsNone(metadata["input_frames"][0]["timestamp_sec"])

    def test_save_window_result_writes_flow_visualization(self):
        conf = np.zeros((2, 8, 8, 1), dtype=np.float32)
        conf[0] = -1000.0
        conf[1] = 1000.0
        arrays = {
            "local_points": np.zeros((2, 8, 8, 3), dtype=np.float32),
            "conf": conf,
            "motion": conf,
            "flow": np.ones((2, 8, 8, 2), dtype=np.float32),
        }
        metadata = {"window_index": 0, "input_frames": [], "predicted_slots": []}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            window_dir = save_window_result(
                output_dir,
                window_index=0,
                arrays=arrays,
                metadata=metadata,
                save_npz=True,
                save_viz=True,
            )

            self.assertTrue((window_dir / "predictions.npz").exists())
            self.assertTrue((window_dir / "metadata.json").exists())
            self.assertTrue((window_dir / "depth" / "000.png").exists())
            self.assertTrue((window_dir / "confidence" / "001.png").exists())
            self.assertTrue((window_dir / "motion" / "001.png").exists())
            self.assertTrue((window_dir / "flow" / "001.png").exists())


class CheckpointInspectionTests(unittest.TestCase):
    def test_uri_style_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local filesystem path"):
            inspect_checkpoint("abc://example/checkpoint.pt")

    def test_model_config_infers_optional_segmentation_head(self):
        state_dict = {
            "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            "segmentation_head.proj.weight": torch.zeros(14 * 14 * 7, 1024),
        }
        config = model_config_from_checkpoint({"MODEL": {"M": 5, "N": 2}}, state_dict)

        self.assertEqual(config.m, 5)
        self.assertEqual(config.n, 2)
        self.assertTrue(config.use_segmentation_head)
        self.assertEqual(config.segmentation_num_classes, 7)
        self.assertTrue(checkpoint_has_autoregressive_state(state_dict))
        self.assertFalse(checkpoint_looks_multiview({}, state_dict))

    def test_model_config_infers_architecture_from_state_and_args(self):
        state_dict = {
            "decoder.0.norm1.weight": torch.zeros(768),
            "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(1536),
            "autoregressive_transformer.blocks.3.norm1.weight": torch.zeros(1536),
            "point_head.input_proj.weight": torch.zeros(256, 3, 1, 1),
        }
        config = model_config_from_checkpoint(
            {"m": 2, "n": 5, "ar_n_heads": 8, "ar_dropout": 0.2},
            state_dict,
        )

        self.assertEqual(config.m, 2)
        self.assertEqual(config.n, 5)
        self.assertEqual(config.decoder_size, "base")
        self.assertEqual(config.ar_n_heads, 8)
        self.assertEqual(config.ar_n_layers, 4)
        self.assertEqual(config.ar_dropout, 0.2)
        self.assertEqual(config.point_head_type, "refined")

    def test_multiview_detection_from_config_and_state(self):
        state_dict = {"autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048)}
        self.assertTrue(checkpoint_looks_multiview({"MODEL": {"ARCHITECTURE": "MultiViewPi3V3"}}, state_dict))
        self.assertTrue(checkpoint_looks_multiview({"MULTIVIEW": {"ENABLE": True, "NUM_CAMERAS": 3}}, state_dict))
        self.assertFalse(
            checkpoint_looks_multiview(
                {
                    "MODEL": {"ARCHITECTURE": "AutoregressivePi3"},
                    "MULTIVIEW": {"ENABLE": False, "NUM_CAMERAS": 5},
                },
                state_dict,
            )
        )
        self.assertTrue(checkpoint_looks_multiview({}, {**state_dict, "scale_token": torch.zeros(1, 1, 1024)}))

    def test_inspect_checkpoint_does_not_require_full_model(self):
        checkpoint = {
            "epoch": 4,
            "model_state_dict": {
                "module.autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
                "module.segmentation_head.proj.weight": torch.zeros(14 * 14 * 7, 1024),
            },
            "config": {"MODEL": {"M": 3, "N": 3}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertTrue(inspection.is_supported_single_view)
            self.assertEqual(inspection.checkpoint_info["epoch"], 4)
            self.assertEqual(inspection.state_dict_key_count, 2)
            self.assertTrue(inspection.optional_heads["segmentation"])

    def test_inspect_checkpoint_metadata_is_json_safe(self):
        checkpoint = {
            "epoch": torch.tensor(4),
            "global_step": torch.tensor([10, 11]),
            "best_loss": float("nan"),
            "val_loss": {"current": torch.tensor(1.25)},
            "best_val_loss": np.float64(0.75),
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": {"MODEL": {"M": 1, "N": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            payload = inspection.to_dict()
            json.dumps(payload, allow_nan=False)
            self.assertEqual(payload["checkpoint_info"]["epoch"], 4)
            self.assertEqual(payload["checkpoint_info"]["global_step"], [10, 11])
            self.assertIsNone(payload["checkpoint_info"]["best_loss"])
            self.assertEqual(payload["checkpoint_info"]["val_loss"], {"current": 1.25})
            self.assertEqual(payload["checkpoint_info"]["best_val_loss"], 0.75)

    def test_inspect_checkpoint_reads_args_payload(self):
        checkpoint = {
            "model_state_dict": {
                "decoder.0.norm1.weight": torch.zeros(1024),
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
                "autoregressive_transformer.blocks.1.norm1.weight": torch.zeros(2048),
            },
            "args": {"m": 4, "n": 2, "ar_n_heads": 8},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertEqual(inspection.model_config.m, 4)
            self.assertEqual(inspection.model_config.n, 2)
            self.assertEqual(inspection.model_config.ar_n_heads, 8)
            self.assertEqual(inspection.model_config.ar_n_layers, 2)

    def test_inspect_checkpoint_reads_argparse_namespace_payload(self):
        checkpoint = {
            "model_state_dict": {
                "decoder.0.norm1.weight": torch.zeros(1024),
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
                "autoregressive_transformer.blocks.1.norm1.weight": torch.zeros(2048),
            },
            "args": argparse.Namespace(m=4, n=2, ar_n_heads=8),
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertEqual(inspection.model_config.m, 4)
            self.assertEqual(inspection.model_config.n, 2)
            self.assertEqual(inspection.model_config.ar_n_heads, 8)
            self.assertEqual(inspection.model_config.ar_n_layers, 2)

    def test_inspect_checkpoint_reads_yacs_config_payload(self):
        from yacs.config import CfgNode as CN

        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": CN({"MODEL": {"M": 2, "N": 0}}),
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertEqual(inspection.model_config.m, 2)
            self.assertEqual(inspection.model_config.n, 0)
            self.assertTrue(inspection.is_supported_single_view)

    def test_checkpoint_loader_rejects_unsupported_pickled_payloads(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "payload": UnsupportedPickledPayload(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "safe weights-only loader"):
                load_checkpoint_file(checkpoint_path)

    def test_inspect_checkpoint_reports_unsupported_config(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": {"MODEL": {"M": 1, "N": 0, "ENCODER_NAME": "dinov3"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertFalse(inspection.is_supported_single_view)
            self.assertIn("Unsupported encoder_name", inspection.configuration_error)

    def test_inspect_checkpoint_rejects_too_many_total_frames(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": {"MODEL": {"M": 12, "N": 4}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            inspection = inspect_checkpoint(checkpoint_path)
            self.assertFalse(inspection.is_supported_single_view)
            self.assertIn("m + n must be <= 15", inspection.configuration_error)


class CheckpointLoadingTests(unittest.TestCase):
    def test_partial_checkpoint_load_is_rejected(self):
        checkpoint = {
            "model_state_dict": {
                "decoder.0.norm1.weight": torch.zeros(384),
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(768),
            },
            "config": {"MODEL": {"M": 1, "N": 0, "AR_N_LAYERS": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "did not fully load"):
                load_model_from_checkpoint(checkpoint_path, device="cpu")


class CliTests(unittest.TestCase):
    def test_multiview_checkpoint_rejection_has_no_traceback(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
                "scale_token": torch.zeros(1, 1, 1024),
            },
            "config": {"MODEL": {"M": 1, "N": 0}, "MULTIVIEW": {"NUM_CAMERAS": 3}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "frames"
            image_dir.mkdir()
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "frame_000.png")
            checkpoint_path = root / "checkpoint.pt"
            output_dir = root / "out"
            torch.save(checkpoint, checkpoint_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lfg.cli",
                    str(image_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            combined_output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error: This checkpoint looks like a multi-view", combined_output)
            self.assertNotIn("Traceback", combined_output)
            self.assertFalse(output_dir.exists())

    def test_uri_style_output_dir_rejection_has_no_traceback(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": {"MODEL": {"M": 1, "N": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "frames"
            image_dir.mkdir()
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "frame_000.png")
            checkpoint_path = root / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lfg.cli",
                    str(image_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--inspect-only",
                    "--output-dir",
                    "abc://example/out",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            combined_output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error: Output directory must be a local filesystem path", combined_output)
            self.assertNotIn("Traceback", combined_output)

    def test_inspect_only_cli_writes_metadata_without_model_load(self):
        checkpoint = {
            "model_state_dict": {
                "autoregressive_transformer.blocks.0.norm1.weight": torch.zeros(2048),
            },
            "config": {"MODEL": {"M": 1, "N": 0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "frames"
            image_dir.mkdir()
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "frame_000.png")
            checkpoint_path = root / "checkpoint.pt"
            output_dir = root / "out"
            torch.save(checkpoint, checkpoint_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "infer.py"),
                    str(image_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--inspect-only",
                    "--decoder-size",
                    "base",
                    "--ar-n-heads",
                    "8",
                    "--ar-n-layers",
                    "4",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"is_supported_single_view": true', result.stdout)
            self.assertIn('"decoder_size": "base"', result.stdout)
            self.assertIn('"ar_n_heads": 8', result.stdout)
            self.assertIn('"ar_n_layers": 4', result.stdout)
            self.assertTrue((output_dir / "run_metadata.json").exists())
            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_inspection"]["kind"], "image_directory")
            self.assertEqual(metadata["input_inspection"]["sampled_frame_count"], 1)
            self.assertEqual(metadata["num_input_frames"], 1)


if __name__ == "__main__":
    unittest.main()
