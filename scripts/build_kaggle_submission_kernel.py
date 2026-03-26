from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "birdclef2026"
KERNEL_DIR = REPO_ROOT / "kaggle" / "birdclef_2026_submit_cpu"
RUN_FILE = KERNEL_DIR / "run.py"
METADATA_FILE = KERNEL_DIR / "kernel-metadata.json"
CHECKPOINT_DATASET = "muditjain1729/birdclef-2026-transformer-sed-checkpoint"
KERNEL_ID = "muditjain1729/birdclef-2026-transformer-sed-submit"
KERNEL_TITLE = "BirdCLEF 2026 ConvNeXt RoPE Submit"
MODULE_FILES = [
    "__init__.py",
    "config.py",
    "audio.py",
    "data.py",
    "kaggle_api.py",
    "model.py",
    "train.py",
    "infer.py",
]


def build_module_payload() -> str:
    payload = {
        module_name: (SRC_DIR / module_name).read_text()
        for module_name in MODULE_FILES
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_metadata() -> str:
    payload = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [CHECKPOINT_DATASET],
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    return json.dumps(payload, indent=2)


def render_run_file(module_payload: str) -> str:
    return f"""from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


INFER_OUTPUT = Path("/kaggle/working/artifacts/infer")
FINAL_SUBMISSION = Path("/kaggle/working/submission.csv")
PACKAGE_ROOT = Path("/kaggle/working/_kernel_pkg")
MODULE_SOURCES = json.loads(r'''{module_payload}''')
CHECKPOINT_DATASET = "{CHECKPOINT_DATASET.split('/')[-1]}"


def detect_data_root() -> Path:
    input_root = Path("/kaggle/input")
    candidates: list[Path] = []
    for path in sorted(input_root.glob("*")):
        if path.is_dir():
            candidates.append(path)
            candidates.extend(sorted(child for child in path.glob("*") if child.is_dir()))
    for candidate in candidates:
        has_submission = (candidate / "sample_submission.csv").exists()
        has_soundscapes = (candidate / "test_soundscapes").exists()
        if has_submission and has_soundscapes:
            print(f"Using competition data from: {{candidate}}")
            return candidate
    available = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate BirdCLEF data under /kaggle/input. "
        f"Scanned:\\n{{available}}"
    )


def detect_checkpoint_path() -> Path:
    input_root = Path("/kaggle/input")
    preferred_root = input_root / CHECKPOINT_DATASET
    preferred = preferred_root / "best_model.pt"
    if preferred.exists():
        print(f"Using checkpoint from: {{preferred}}")
        return preferred
    matches = sorted(input_root.rglob("best_model.pt"))
    if matches:
        print(f"Using checkpoint from: {{matches[0]}}")
        return matches[0]
    raise FileNotFoundError("Unable to locate best_model.pt under /kaggle/input")


def materialize_package() -> None:
    package_dir = PACKAGE_ROOT / "birdclef2026"
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, source in MODULE_SOURCES.items():
        target = package_dir / relative_path
        target.write_text(source)


materialize_package()
sys.path.insert(0, str(PACKAGE_ROOT))
DATA_ROOT = detect_data_root()
CHECKPOINT_PATH = detect_checkpoint_path()

from birdclef2026.infer import main as infer_main  # noqa: E402


def run_infer() -> Path:
    sys.argv = [
        "birdclef-infer",
        "--data-root",
        str(DATA_ROOT),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--output-dir",
        str(INFER_OUTPUT),
        "--window-seconds",
        "5",
        "--batch-size",
        "8",
    ]
    submission = infer_main()
    shutil.copy2(submission, FINAL_SUBMISSION)
    return FINAL_SUBMISSION


if __name__ == "__main__":
    submission_path = run_infer()
    print(f"submission={{submission_path}}")
"""


def main() -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_FILE.write_text(render_run_file(build_module_payload()))
    METADATA_FILE.write_text(render_metadata())
    print(KERNEL_DIR)


if __name__ == "__main__":
    main()
