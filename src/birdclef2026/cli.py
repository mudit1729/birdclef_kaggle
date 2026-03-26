from __future__ import annotations

import argparse
from pathlib import Path

from birdclef2026.config import BASELINE_ROOT, COMPETITION_ROOT
from birdclef2026.infer import main as infer_main_impl
from birdclef2026.kaggle_api import (
    DEFAULT_COMPETITION,
    DEFAULT_KERNEL,
    download_competition,
    download_kernel_output,
    get_kernel_status,
    pull_baseline,
    submit_code_competition,
    submit_competition,
    submit_kernel_output,
)
from birdclef2026.train import main as train_main_impl


def pull_baseline_main() -> None:
    parser = argparse.ArgumentParser(description="Pull the baseline Kaggle notebook.")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--destination", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--metadata-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(
        pull_baseline(
            kernel=args.kernel,
            destination=args.destination,
            metadata_only=args.metadata_only,
        )
    )


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download BirdCLEF 2026 competition data.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--destination", type=Path, default=COMPETITION_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        download_competition(
            competition=args.competition,
            destination=args.destination,
            force=args.force,
        )
    )


def submit_main() -> None:
    parser = argparse.ArgumentParser(description="Submit a BirdCLEF 2026 solution to Kaggle.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    print(
        submit_competition(
            file_path=args.file,
            message=args.message,
            competition=args.competition,
        )
    )


def submit_kernel_main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for a Kaggle kernel to finish, download submission.csv, and submit it."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=60 * 60)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    destination = args.destination
    if args.download_only:
        print(get_kernel_status(args.kernel))
        print(
            download_kernel_output(
                kernel=args.kernel,
                destination=destination or Path("artifacts/kernel_output"),
            )
        )
        return
    print(
        submit_kernel_output(
            kernel=args.kernel,
            message=args.message,
            competition=args.competition,
            destination=destination or Path("artifacts/kernel_output"),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    )


def submit_code_main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a Kaggle code-competition notebook version."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-version", type=int, default=None)
    parser.add_argument("--file-name", default="submission.csv")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    print(
        submit_code_competition(
            file_name=args.file_name,
            message=args.message,
            kernel=args.kernel,
            competition=args.competition,
            kernel_version=args.kernel_version,
        )
    )


def train_main() -> None:
    checkpoint = train_main_impl()
    print(checkpoint)


def infer_main() -> None:
    submission = infer_main_impl()
    print(submission)
