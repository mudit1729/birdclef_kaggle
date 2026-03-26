from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from birdclef2026.config import ARTIFACTS_ROOT, REPO_ROOT

DATASET_DIR = REPO_ROOT / "kaggle" / "birdclef_2026_checkpoint_dataset"
DEFAULT_CHECKPOINT = ARTIFACTS_ROOT / "kernel_output" / "artifacts" / "train" / "best_model.pt"
DEFAULT_HISTORY = ARTIFACTS_ROOT / "kernel_output" / "artifacts" / "train" / "history.json"
DATASET_ID = "muditjain1729/birdclef-2026-transformer-sed-checkpoint"
DATASET_TITLE = "BirdCLEF 2026 Transformer SED Checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a Kaggle dataset folder containing the trained BirdCLEF checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--title", default=DATASET_TITLE)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (args.output_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.copy2(args.checkpoint, args.output_dir / "best_model.pt")
    if args.history.exists():
        shutil.copy2(args.history, args.output_dir / "history.json")
    print(args.output_dir)
    return args.output_dir


if __name__ == "__main__":
    main()
