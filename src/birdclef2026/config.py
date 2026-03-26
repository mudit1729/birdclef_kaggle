from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
COMPETITION_ROOT = DATA_ROOT / "birdclef-2026"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
BASELINE_ROOT = REPO_ROOT / "external" / "kaggle-baseline"


@dataclass(slots=True)
class CompetitionLayout:
    root: Path
    train_csv: Path
    train_audio: Path
    sample_submission: Path
    test_soundscapes: Path


def infer_layout(root: Path | None = None) -> CompetitionLayout:
    competition_root = root or COMPETITION_ROOT
    return CompetitionLayout(
        root=competition_root,
        train_csv=competition_root / "train.csv",
        train_audio=competition_root / "train_audio",
        sample_submission=competition_root / "sample_submission.csv",
        test_soundscapes=competition_root / "test_soundscapes",
    )
