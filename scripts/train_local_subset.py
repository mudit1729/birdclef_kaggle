from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from birdclef2026.config import ARTIFACTS_ROOT, COMPETITION_ROOT
from birdclef2026.kaggle_api import DEFAULT_COMPETITION, download_competition_files
from birdclef2026.train import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small BirdCLEF subset and train locally on the transformer model."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--data-root", type=Path, default=COMPETITION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT / "local_subset_train")
    parser.add_argument("--architecture", default="efficientnet_transformer_sed")
    parser.add_argument("--backbone", default="convnext_nano")
    parser.add_argument("--label-count", type=int, default=4)
    parser.add_argument("--clips-per-label", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument(
        "--transformer-pooling",
        choices=["clip_attention", "attention", "mean", "max"],
        default="attention",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def select_subset(
    train_csv: Path,
    label_count: int,
    clips_per_label: int,
) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    with train_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["primary_label"].strip()
            filename = row["filename"].strip()
            if label and filename:
                by_label[label].append(filename)

    ranked_labels = sorted(by_label, key=lambda label: (-len(by_label[label]), label))
    chosen_labels: list[str] = []
    chosen_files: list[str] = []
    for label in ranked_labels:
        files = by_label[label][:clips_per_label]
        if len(files) < 2:
            continue
        chosen_labels.append(label)
        chosen_files.extend(files)
        if len(chosen_labels) >= label_count:
            break
    if len(chosen_labels) < label_count:
        raise RuntimeError(
            f"Unable to find {label_count} labels with at least 2 clips in {train_csv}."
        )
    return chosen_labels, chosen_files


def main() -> Path:
    args = parse_args()
    train_csv = args.data_root / "train.csv"
    sample_submission = args.data_root / "sample_submission.csv"
    if not train_csv.exists() or not sample_submission.exists():
        raise FileNotFoundError(
            "Expected train.csv and sample_submission.csv under the local competition data root."
        )

    labels, files = select_subset(
        train_csv=train_csv,
        label_count=args.label_count,
        clips_per_label=args.clips_per_label,
    )
    downloaded = download_competition_files(
        files=[f"train_audio/{file}" for file in files],
        competition=args.competition,
        destination=args.data_root,
        force=args.force_download,
    )
    print(f"labels={labels}")
    print(f"downloaded_clips={len(downloaded)}")

    sys.argv = [
        "birdclef-train",
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.output_dir),
        "--architecture",
        args.architecture,
        "--backbone",
        args.backbone,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        "0",
        "--clip-seconds",
        "5",
        "--image-height",
        str(args.image_height),
        "--image-width",
        str(args.image_width),
        "--n-mels",
        "128",
        "--validation-fraction",
        str(args.validation_fraction),
        "--transformer-dim",
        str(args.transformer_dim),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-pooling",
        args.transformer_pooling,
        "--max-samples",
        str(len(files)),
        "--no-pretrained",
    ]
    checkpoint = train_main()
    print(checkpoint)
    return checkpoint


if __name__ == "__main__":
    main()
