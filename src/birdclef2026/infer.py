from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from birdclef2026.audio import crop_or_pad, load_audio, waveform_to_image
from birdclef2026.config import ARTIFACTS_ROOT, COMPETITION_ROOT
from birdclef2026.data import build_submission_map, load_layout
from birdclef2026.kaggle_api import DEFAULT_COMPETITION, ensure_competition_data
from birdclef2026.model import build_model, combine_primary_secondary_probabilities
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
    parser.add_argument("--tta-shifts", type=int, default=0,
                        help="Number of TTA temporal shifts (0=disabled, 1=+/-2.5s)")
    parser.add_argument("--topn-postprocess", action="store_true",
                        help="Apply TopN file-level postprocessing")
    parser.add_argument("--temporal-smoothing", action="store_true",
                        help="Apply temporal smoothing across adjacent segments")
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


def topn_postprocess(
    file_predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Multiply each segment's prob by the file-level max of that class.

    This boosts species that are consistently detected and suppresses spurious detections.
    From BirdCLEF 2025 2nd place solution (+0.011 LB).
    """
    if not file_predictions:
        return file_predictions
    row_ids = list(file_predictions.keys())
    probs = np.stack([file_predictions[rid] for rid in row_ids])
    file_max = probs.max(axis=0, keepdims=True)
    probs = probs * file_max
    return {rid: probs[i] for i, rid in enumerate(row_ids)}


def temporal_smoothing(
    ordered_row_ids: list[str],
    predictions: dict[str, np.ndarray],
    kernel: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Apply temporal smoothing across adjacent segments within a soundscape file.

    From BirdCLEF 2025 top solutions (+0.010 LB).
    """
    if len(ordered_row_ids) <= 1:
        return predictions
    if kernel is None:
        kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
    half = len(kernel) // 2
    probs = np.stack([predictions[rid] for rid in ordered_row_ids])
    smoothed = np.zeros_like(probs)
    for i in range(len(probs)):
        total_weight = 0.0
        for j, w in enumerate(kernel):
            idx = i + j - half
            if 0 <= idx < len(probs):
                smoothed[i] += w * probs[idx]
                total_weight += w
        if total_weight > 0:
            smoothed[i] /= total_weight
    return {rid: smoothed[i] for i, rid in enumerate(ordered_row_ids)}


def run_batch_inference(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
) -> list[np.ndarray]:
    """Run model inference on a batch of images and return probability arrays."""
    if not images:
        return []
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(batch)
    if isinstance(outputs, dict):
        probs = combine_primary_secondary_probabilities(
            outputs["primary_logits"],
            outputs["secondary_logits"],
        ).detach().cpu().numpy()
    else:
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
    return [probs[i] for i in range(probs.shape[0])]


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
    sample_rate = int(config["sample_rate"])

    device = select_device()
    model = build_model(
        num_classes=len(labels),
        architecture=config.get("architecture", "efficientnet_classifier"),
        backbone=config["backbone"],
        pretrained=False,
        classifier_head_mode=config.get("classifier_head_mode", "single"),
        transformer_dim=int(config.get("transformer_dim", 256)),
        transformer_heads=int(config.get("transformer_heads", 8)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_pooling=config.get("transformer_pooling", "clip_attention"),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mel_kwargs = dict(
        sample_rate=sample_rate,
        image_height=image_height,
        image_width=image_width,
        n_mels=int(config["n_mels"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        fmin=int(config["fmin"]),
        fmax=int(config["fmax"]),
    )

    # TTA shift offsets in seconds (0 = no shift, plus +/- half-window shifts)
    tta_offsets = [0.0]
    if args.tta_shifts >= 1:
        half_window = args.window_seconds / 2.0
        tta_offsets = [-half_window, 0.0, half_window]

    submission = pd.read_csv(layout.sample_submission)
    row_mapping = build_submission_map(submission)
    audio_files = {path.stem: path for path in layout.test_soundscapes.rglob("*") if path.is_file()}

    predictions: dict[str, np.ndarray] = {}
    for stem, row_specs in tqdm(row_mapping.items()):
        audio_path = audio_files.get(stem)
        if audio_path is None:
            continue
        waveform = load_audio(audio_path, sample_rate)
        sorted_specs = sorted(row_specs)
        file_row_ids: list[str] = []

        batch_images: list[torch.Tensor] = []
        batch_meta: list[tuple[str, int]] = []  # (row_id, tta_idx)

        for seconds, row_id in sorted_specs:
            file_row_ids.append(row_id)
            for tta_idx, offset in enumerate(tta_offsets):
                shifted_end = max(args.window_seconds, seconds + offset)
                chunk = chunk_from_end(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    end_second=int(shifted_end),
                    window_seconds=args.window_seconds,
                )
                image = waveform_to_image(waveform=chunk, **mel_kwargs)
                batch_images.append(image)
                batch_meta.append((row_id, tta_idx))

                if len(batch_images) == args.batch_size:
                    probs_list = run_batch_inference(model, batch_images, device)
                    for (rid, tidx), prob in zip(batch_meta, probs_list, strict=True):
                        key = f"{rid}__tta{tidx}"
                        predictions[key] = prob
                    batch_images = []
                    batch_meta = []

        if batch_images:
            probs_list = run_batch_inference(model, batch_images, device)
            for (rid, tidx), prob in zip(batch_meta, probs_list, strict=True):
                key = f"{rid}__tta{tidx}"
                predictions[key] = prob

        # Average TTA predictions per row_id
        file_preds: dict[str, np.ndarray] = {}
        for row_id in file_row_ids:
            tta_probs = [
                predictions[f"{row_id}__tta{i}"]
                for i in range(len(tta_offsets))
                if f"{row_id}__tta{i}" in predictions
            ]
            if tta_probs:
                file_preds[row_id] = np.mean(tta_probs, axis=0)

        # TopN postprocessing: multiply by file-level max per class
        if args.topn_postprocess:
            file_preds = topn_postprocess(file_preds)

        # Temporal smoothing across adjacent segments
        if args.temporal_smoothing:
            file_preds = temporal_smoothing(file_row_ids, file_preds)

        for row_id, prob in file_preds.items():
            predictions[row_id] = prob

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
