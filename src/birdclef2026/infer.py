from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from birdclef2026.audio import crop_or_pad, load_audio, waveform_to_image
from birdclef2026.config import ARTIFACTS_ROOT, COMPETITION_ROOT
from birdclef2026.data import build_submission_map, load_layout
from birdclef2026.kaggle_api import DEFAULT_COMPETITION, ensure_competition_data
from birdclef2026.model import build_model
from birdclef2026.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BirdCLEF 2026 inference and build submission."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--data-root", type=Path, default=COMPETITION_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT / "infer")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--download-if-missing", action="store_true")
    return parser.parse_args()


def chunk_from_end(
    waveform,
    sample_rate: int,
    end_second: int,
    window_seconds: float,
):
    end_idx = int(end_second * sample_rate)
    start_idx = max(0, end_idx - int(window_seconds * sample_rate))
    chunk = waveform[start_idx:end_idx]
    return crop_or_pad(
        chunk,
        sample_rate=sample_rate,
        clip_seconds=window_seconds,
        random_crop=False,
    )


def main() -> Path:
    args = parse_args()
    if args.download_if_missing:
        ensure_competition_data(competition=args.competition, destination=args.data_root)
    layout = load_layout(args.data_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    labels = checkpoint["labels"]
    image_height = int(config.get("image_height", config.get("image_size", 224)))
    image_width = int(config.get("image_width", config.get("image_size", 224)))

    device = select_device()
    model = build_model(
        num_classes=len(labels),
        architecture=config.get("architecture", "efficientnet_classifier"),
        backbone=config["backbone"],
        pretrained=False,
        transformer_dim=int(config.get("transformer_dim", 256)),
        transformer_heads=int(config.get("transformer_heads", 8)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    submission = pd.read_csv(layout.sample_submission)
    row_mapping = build_submission_map(submission)
    audio_files = {path.stem: path for path in layout.test_soundscapes.rglob("*") if path.is_file()}

    predictions: dict[str, torch.Tensor] = {}
    for stem, row_specs in tqdm(row_mapping.items()):
        audio_path = audio_files.get(stem)
        if audio_path is None:
            continue
        waveform = load_audio(audio_path, int(config["sample_rate"]))
        batch_images = []
        row_ids = []
        for seconds, row_id in sorted(row_specs):
            chunk = chunk_from_end(
                waveform=waveform,
                sample_rate=int(config["sample_rate"]),
                end_second=seconds,
                window_seconds=args.window_seconds,
            )
            image = waveform_to_image(
                waveform=chunk,
                sample_rate=int(config["sample_rate"]),
                image_height=image_height,
                image_width=image_width,
                n_mels=int(config["n_mels"]),
                n_fft=int(config["n_fft"]),
                hop_length=int(config["hop_length"]),
                fmin=int(config["fmin"]),
                fmax=int(config["fmax"]),
            )
            batch_images.append(image)
            row_ids.append(row_id)
            if len(batch_images) == args.batch_size:
                batch = torch.stack(batch_images).to(device)
                probs = torch.sigmoid(model(batch)).detach().cpu()
                for local_row_id, prob in zip(row_ids, probs, strict=True):
                    predictions[local_row_id] = prob
                batch_images = []
                row_ids = []
        if batch_images:
            batch = torch.stack(batch_images).to(device)
            probs = torch.sigmoid(model(batch)).detach().cpu()
            for local_row_id, prob in zip(row_ids, probs, strict=True):
                predictions[local_row_id] = prob

    label_to_index = {label: index for index, label in enumerate(labels)}
    for row_id in submission["row_id"].astype(str):
        prob = predictions.get(row_id)
        if prob is None:
            for label in labels:
                if label in submission.columns:
                    submission.loc[submission["row_id"] == row_id, label] = 0.0
            continue
        for label in labels:
            if label in submission.columns:
                submission.loc[submission["row_id"] == row_id, label] = float(
                    prob[label_to_index[label]]
                )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    submission_path = main()
    print(submission_path)
