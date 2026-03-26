from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from birdclef2026.data import Example
from birdclef2026.kaggle_api import parse_kernel_status
from birdclef2026.model import build_model
from birdclef2026.train import (
    compute_class_focal_alpha,
    compute_label_counts,
    normalized_inverse_frequency_weights,
)


def create_tone(
    path: Path,
    frequency: float,
    seconds: float = 6.0,
    sample_rate: int = 32_000,
) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)


def build_fake_dataset(root: Path) -> Path:
    data_root = root / "birdclef-2026"
    train_audio = data_root / "train_audio"
    test_soundscapes = data_root / "test_soundscapes"

    create_tone(train_audio / "bird_a" / "a_0.ogg", 440)
    create_tone(train_audio / "bird_a" / "a_1.ogg", 450)
    create_tone(train_audio / "bird_b" / "b_0.ogg", 660)
    create_tone(train_audio / "bird_b" / "b_1.ogg", 670)
    create_tone(test_soundscapes / "soundscape_1.ogg", 440, seconds=10.0)

    train_df = pd.DataFrame(
        [
            {"filename": "a_0.ogg", "primary_label": "bird_a", "secondary_labels": "[]"},
            {"filename": "a_1.ogg", "primary_label": "bird_a", "secondary_labels": "[]"},
            {"filename": "b_0.ogg", "primary_label": "bird_b", "secondary_labels": "[]"},
            {"filename": "b_1.ogg", "primary_label": "bird_b", "secondary_labels": "[]"},
        ]
    )
    submission_df = pd.DataFrame(
        [
            {"row_id": "soundscape_1_5", "bird_a": 0.0, "bird_b": 0.0},
            {"row_id": "soundscape_1_10", "bird_a": 0.0, "bird_b": 0.0},
        ]
    )
    train_df.to_csv(data_root / "train.csv", index=False)
    submission_df.to_csv(data_root / "sample_submission.csv", index=False)
    return data_root


def run_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd / "src")
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )


def test_smoke_train_and_infer(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = build_fake_dataset(tmp_path)
    output_dir = tmp_path / "artifacts"

    train_result = run_module(
        [
            "-m",
            "birdclef2026.train",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir / "train"),
            "--architecture",
            "efficientnet_transformer_sed",
            "--backbone",
            "convnext_atto",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "128",
            "--transformer-dim",
            "64",
            "--transformer-heads",
            "4",
            "--transformer-layers",
            "1",
            "--num-workers",
            "0",
            "--no-pretrained",
        ],
        cwd=repo_root,
    )
    checkpoint_path = Path(train_result.stdout.strip().splitlines()[-1])
    assert checkpoint_path.exists()
    history = json.loads((output_dir / "train" / "history.json").read_text())
    assert history
    assert {
        "valid_accuracy",
        "valid_precision",
        "valid_recall",
        "valid_f1",
        "valid_precision_micro",
        "valid_recall_micro",
        "valid_f1_micro",
    } <= history[0].keys()
    class_alpha = json.loads((output_dir / "train" / "class_alpha.json").read_text())
    assert class_alpha
    assert {
        "label",
        "primary_count",
        "secondary_count",
        "combined_count",
        "primary_weight",
        "secondary_alpha",
        "combined_alpha",
    } <= class_alpha[0].keys()
    thresholds = json.loads((output_dir / "train" / "validation_thresholds.json").read_text())
    assert thresholds["thresholds"]

    infer_result = run_module(
        [
            "-m",
            "birdclef2026.infer",
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir / "infer"),
        ],
        cwd=repo_root,
    )
    submission_path = Path(infer_result.stdout.strip().splitlines()[-1])
    assert submission_path.exists()
    submission = pd.read_csv(submission_path)
    assert list(submission.columns) == ["row_id", "bird_a", "bird_b"]
    assert submission.shape[0] == 2


def test_build_model_variants() -> None:
    inputs = torch.randn(2, 1, 128, 128)
    htsat_inputs = torch.randn(2, 1, 224, 224)

    classifier = build_model(
        num_classes=3,
        architecture="efficientnet_classifier",
        backbone="convnext_atto",
        pretrained=False,
    )
    classifier_dual = build_model(
        num_classes=3,
        architecture="efficientnet_classifier",
        backbone="convnext_atto",
        classifier_head_mode="dual",
        pretrained=False,
    )
    transformer = build_model(
        num_classes=3,
        architecture="efficientnet_transformer_sed",
        backbone="convnext_atto",
        pretrained=False,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        transformer_pooling="attention",
        dropout=0.1,
    )
    transformer_mean = build_model(
        num_classes=3,
        architecture="efficientnet_transformer_sed",
        backbone="convnext_atto",
        pretrained=False,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        transformer_pooling="mean",
        dropout=0.1,
    )
    transformer_max = build_model(
        num_classes=3,
        architecture="efficientnet_transformer_sed",
        backbone="convnext_atto",
        pretrained=False,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        transformer_pooling="max",
        dropout=0.1,
    )
    transformer_legacy = build_model(
        num_classes=3,
        architecture="efficientnet_transformer_sed",
        backbone="convnext_atto",
        pretrained=False,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        transformer_pooling="clip_attention",
        dropout=0.1,
    )
    htsat = build_model(
        num_classes=3,
        architecture="htsat_token_semantic",
        backbone="swin_tiny_patch4_window7_224",
        pretrained=False,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        dropout=0.1,
    )

    assert classifier(inputs).shape == (2, 3)
    dual_outputs = classifier_dual(inputs)
    assert set(dual_outputs) == {"primary_logits", "secondary_logits"}
    assert dual_outputs["primary_logits"].shape == (2, 3)
    assert dual_outputs["secondary_logits"].shape == (2, 3)
    assert transformer(inputs).shape == (2, 3)
    assert transformer_mean(inputs).shape == (2, 3)
    assert transformer_max(inputs).shape == (2, 3)
    assert transformer_legacy(inputs).shape == (2, 3)
    assert htsat(htsat_inputs).shape == (2, 3)


def test_parse_kernel_status() -> None:
    status = parse_kernel_status(
        'muditjain1729/birdclef-2026-eb0-gpu-baseline has status "KernelWorkerStatus.RUNNING"'
    )
    assert status == "KernelWorkerStatus.RUNNING"


def test_compute_class_focal_alpha_weights_rare_classes_more() -> None:
    examples = [
        Example(audio_path=Path("common_0.ogg"), labels=["common"], primary_label="common"),
        Example(audio_path=Path("common_1.ogg"), labels=["common"], primary_label="common"),
        Example(
            audio_path=Path("mixed.ogg"),
            labels=["common", "rare"],
            primary_label="rare",
        ),
    ]
    counts = compute_label_counts(examples, ["rare", "common"], mode="combined")
    weights = normalized_inverse_frequency_weights(counts)
    alpha = compute_class_focal_alpha(counts, alpha_min=0.05, alpha_max=0.95)
    assert 0.05 <= float(alpha.min()) <= float(alpha.max()) <= 0.95
    assert weights[0] > weights[1]
    assert alpha[0] > alpha[1]
