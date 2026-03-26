from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from birdclef2026.config import ARTIFACTS_ROOT, BASELINE_ROOT, COMPETITION_ROOT

DEFAULT_COMPETITION = "birdclef-2026"
DEFAULT_KERNEL = "leonshangguan/faster-eb0-sed-model-inference"


class KaggleAuthError(RuntimeError):
    """Raised when Kaggle credentials are missing or rejected."""


def kaggle_has_credentials() -> bool:
    home = Path.home()
    legacy_file = home / ".kaggle" / "kaggle.json"
    access_token = home / ".kaggle" / "access_token"
    return (
        legacy_file.exists()
        or access_token.exists()
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")))
    )


def _run_kaggle(args: list[str]) -> str:
    if not kaggle_has_credentials():
        raise KaggleAuthError(
            "No Kaggle credentials found. Add ~/.kaggle/access_token or ~/.kaggle/kaggle.json."
        )
    try:
        completed = subprocess.run(
            ["kaggle", *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The `kaggle` CLI is not installed in the active environment.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message = stderr or stdout or "Unknown Kaggle CLI error."
        raise RuntimeError(message) from exc
    return (completed.stdout or "").strip()


def pull_baseline(
    kernel: str = DEFAULT_KERNEL,
    destination: Path = BASELINE_ROOT,
    metadata_only: bool = True,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "pull", kernel, "-p", str(destination)]
    if metadata_only:
        args.append("-m")
    _run_kaggle(args)
    return destination


def download_competition(
    competition: str = DEFAULT_COMPETITION,
    destination: Path = COMPETITION_ROOT,
    unzip: bool = True,
    force: bool = False,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    args = ["competitions", "download", "-c", competition, "-p", str(destination)]
    if unzip:
        args.append("--unzip")
    if force:
        args.append("--force")
    _run_kaggle(args)
    return destination


def download_competition_files(
    files: list[str],
    competition: str = DEFAULT_COMPETITION,
    destination: Path = COMPETITION_ROOT / "train_audio",
    force: bool = False,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    materialized: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="birdclef-kaggle-") as temp_dir:
        temp_root = Path(temp_dir)
        for raw_file in files:
            relative_path = Path(raw_file)
            target = destination / relative_path
            if target.exists() and not force:
                materialized.append(target)
                continue
            _run_kaggle(
                [
                    "competitions",
                    "download",
                    "-c",
                    competition,
                    "-f",
                    relative_path.as_posix(),
                    "-p",
                    str(temp_root),
                    "--force",
                ]
            )
            downloaded = next(temp_root.rglob(relative_path.name), None)
            if downloaded is None:
                raise FileNotFoundError(f"Kaggle did not return the requested file: {raw_file}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), target)
            materialized.append(target)
    return materialized


def parse_kernel_status(output: str) -> str:
    match = re.search(r'has status "(?P<status>[^"]+)"', output)
    if match is None:
        raise ValueError(f"Unable to parse kernel status from: {output}")
    return match.group("status")


def get_kernel_status(kernel: str) -> str:
    return parse_kernel_status(_run_kaggle(["kernels", "status", kernel]))


def download_kernel_output(
    kernel: str,
    destination: Path = ARTIFACTS_ROOT / "kernel_output",
    force: bool = True,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel, "-p", str(destination)]
    if force:
        args.append("--force")
    _run_kaggle(args)
    return destination


def wait_for_kernel_completion(
    kernel: str,
    poll_seconds: int = 30,
    timeout_seconds: int = 60 * 60,
) -> str:
    started_at = time.monotonic()
    terminal_errors = {
        "KernelWorkerStatus.ERROR",
        "KernelWorkerStatus.CANCELLED",
    }
    while True:
        status = get_kernel_status(kernel)
        if status == "KernelWorkerStatus.COMPLETE":
            return status
        if status in terminal_errors:
            raise RuntimeError(f"Kernel {kernel} finished with status {status}")
        if time.monotonic() - started_at >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for kernel {kernel}; last status={status}")
        time.sleep(poll_seconds)


def submit_kernel_output(
    kernel: str,
    message: str,
    competition: str = DEFAULT_COMPETITION,
    destination: Path = ARTIFACTS_ROOT / "kernel_output",
    poll_seconds: int = 30,
    timeout_seconds: int = 60 * 60,
) -> str:
    wait_for_kernel_completion(
        kernel=kernel,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    output_dir = download_kernel_output(kernel=kernel, destination=destination, force=True)
    submission_path = output_dir / "submission.csv"
    if not submission_path.exists():
        raise FileNotFoundError(f"Kernel output does not contain submission.csv: {output_dir}")
    return submit_competition(
        file_path=submission_path,
        message=message,
        competition=competition,
    )


def submit_competition(
    file_path: Path,
    message: str,
    competition: str = DEFAULT_COMPETITION,
) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Submission file not found: {file_path}")
    return _run_kaggle(
        ["competitions", "submit", "-c", competition, "-f", str(file_path), "-m", message]
    )


def submit_code_competition(
    file_name: str,
    message: str,
    kernel: str,
    competition: str = DEFAULT_COMPETITION,
    kernel_version: int | None = None,
) -> str:
    if not kaggle_has_credentials():
        raise KaggleAuthError(
            "No Kaggle credentials found. Add ~/.kaggle/access_token or ~/.kaggle/kaggle.json."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    response = api.competition_submit_code(
        file_name=file_name,
        message=message,
        competition=competition,
        kernel=kernel,
        kernel_version=kernel_version,
    )
    payload = response.to_dict() if hasattr(response, "to_dict") else {"response": str(response)}
    return json.dumps(payload, indent=2, sort_keys=True)


def ensure_competition_data(
    competition: str = DEFAULT_COMPETITION,
    destination: Path = COMPETITION_ROOT,
) -> Path:
    required = [
        destination / "train.csv",
        destination / "train_audio",
        destination / "sample_submission.csv",
    ]
    if all(path.exists() for path in required):
        return destination
    return download_competition(competition=competition, destination=destination)
