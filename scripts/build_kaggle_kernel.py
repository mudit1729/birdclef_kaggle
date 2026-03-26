from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "birdclef2026"
KAGGLE_ROOT = REPO_ROOT / "kaggle"
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


@dataclass(frozen=True, slots=True)
class KernelPreset:
    name: str
    directory_name: str
    kernel_id: str
    title: str
    architecture: str
    backbone: str
    epochs: int
    image_height: int
    image_width: int
    transformer_dim: int
    transformer_heads: int
    transformer_layers: int
    gpu_batch_size: int
    cpu_batch_size: int
    gpu_max_samples: int
    cpu_max_samples: int
    gpu_infer_batch_size: int
    cpu_infer_batch_size: int
    # v6 augmentation and training improvements
    loss: str = "focal"
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    random_filter_prob: float = 0.0
    freq_mask_param: int = 0
    time_mask_param: int = 0
    drop_path_rate: float = 0.0
    # v6 inference improvements
    tta_shifts: int = 0
    topn_postprocess: bool = False
    temporal_smoothing: bool = False


PRESETS: dict[str, KernelPreset] = {
    "classifier": KernelPreset(
        name="classifier",
        directory_name="birdclef_2026_classifier_gpu",
        kernel_id="muditjain1729/birdclef-2026-convnext-classifier-gpu",
        title="BirdCLEF 2026 ConvNeXt Classifier GPU",
        architecture="efficientnet_classifier",
        backbone="convnext_nano",
        epochs=10,
        image_height=128,
        image_width=256,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        gpu_batch_size=16,
        cpu_batch_size=4,
        gpu_max_samples=12000,
        cpu_max_samples=1500,
        gpu_infer_batch_size=48,
        cpu_infer_batch_size=8,
    ),
    "transformer": KernelPreset(
        name="transformer",
        directory_name="birdclef_2026_transformer_gpu",
        kernel_id="muditjain1729/birdclef-2026-convnext-rope-xfmr-gpu-v6",
        title="BirdCLEF 2026 ConvNeXt RoPE Xfmr GPU v6",
        architecture="efficientnet_transformer_sed",
        backbone="convnext_nano",
        epochs=25,
        image_height=128,
        image_width=256,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=2,
        gpu_batch_size=12,
        cpu_batch_size=2,
        gpu_max_samples=50000,
        cpu_max_samples=1000,
        gpu_infer_batch_size=32,
        cpu_infer_batch_size=8,
        # v6 improvements
        loss="focal",
        label_smoothing=0.05,
        mixup_alpha=0.5,
        random_filter_prob=0.5,
        freq_mask_param=30,
        time_mask_param=0,
        drop_path_rate=0.1,
        tta_shifts=1,
        topn_postprocess=True,
        temporal_smoothing=True,
    ),
    "htsat": KernelPreset(
        name="htsat",
        directory_name="birdclef_2026_gpu",
        kernel_id="muditjain1729/birdclef-2026-hts-at-gpu-baseline",
        title="BirdCLEF 2026 HTS AT GPU Baseline",
        architecture="htsat_token_semantic",
        backbone="swin_tiny_patch4_window7_224",
        epochs=10,
        image_height=224,
        image_width=224,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=1,
        gpu_batch_size=8,
        cpu_batch_size=2,
        gpu_max_samples=8000,
        cpu_max_samples=1000,
        gpu_infer_batch_size=24,
        cpu_infer_batch_size=8,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle GPU training kernels.")
    parser.add_argument(
        "--preset",
        choices=[*PRESETS.keys(), "all"],
        default="all",
        help="Kernel preset to build. Default builds all presets.",
    )
    return parser.parse_args()


def build_module_payload() -> str:
    payload = {
        module_name: (SRC_DIR / module_name).read_text()
        for module_name in MODULE_FILES
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_run_file(module_payload: str, preset: KernelPreset) -> str:
    return f"""from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


KERNEL_ROOT = Path(__file__).resolve().parent
TRAIN_OUTPUT = Path("/kaggle/working/artifacts/train")
INFER_OUTPUT = Path("/kaggle/working/artifacts/infer")
FINAL_SUBMISSION = Path("/kaggle/working/submission.csv")
PACKAGE_ROOT = Path("/kaggle/working/_kernel_pkg")
MODULE_SOURCES = json.loads(r'''{module_payload}''')
TORCH_BOOTSTRAP_ENV = "BIRDCLEF_TORCH_BOOTSTRAPPED"
NUMPY_VERSION = "2.0.2"
TORCH_VERSION = "2.6.0+cu118"
TORCHVISION_VERSION = "0.21.0+cu118"


def detect_data_root() -> Path:
    input_root = Path("/kaggle/input")
    candidates: list[Path] = []
    for path in sorted(input_root.glob("*")):
        if path.is_dir():
            candidates.append(path)
            candidates.extend(sorted(child for child in path.glob("*") if child.is_dir()))
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "sample_submission.csv").exists():
            print(f"Using competition data from: {{candidate}}")
            return candidate
    available = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate BirdCLEF data under /kaggle/input. "
        f"Scanned:\\n{{available}}"
    )


def torch_supports_active_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    arch = f"sm_{{major}}{{minor}}"
    arch_list_fn = getattr(torch.cuda, "get_arch_list", None)
    if callable(arch_list_fn):
        try:
            supported_arches = arch_list_fn()
        except Exception:
            supported_arches = []
        if supported_arches:
            print(f"Current torch CUDA arches: {{supported_arches}}")
            return arch in supported_arches
    return major >= 7


def ensure_gpu_compatible_torch() -> None:
    if os.environ.get(TORCH_BOOTSTRAP_ENV) == "1":
        return
    if torch_supports_active_gpu():
        return
    print("Installing GPU-compatible PyTorch wheels with a NumPy pin for librosa/numba...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            f"numpy=={{NUMPY_VERSION}}",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            f"torch=={{TORCH_VERSION}}",
            f"torchvision=={{TORCHVISION_VERSION}}",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            f"numpy=={{NUMPY_VERSION}}",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import numpy, torch; "
                "print(f'Bootstrapped numpy={{numpy.__version__}} torch={{torch.__version__}}')"
            ),
        ]
    )
    env = dict(os.environ)
    env[TORCH_BOOTSTRAP_ENV] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_gpu_compatible_torch()

import torch


def cuda_device_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    arch = f"sm_{{major}}{{minor}}"
    arch_list_fn = getattr(torch.cuda, "get_arch_list", None)
    if callable(arch_list_fn):
        try:
            supported_arches = arch_list_fn()
        except Exception:
            supported_arches = []
        if supported_arches:
            return arch in supported_arches
    return major >= 7


def materialize_package() -> None:
    package_dir = PACKAGE_ROOT / "birdclef2026"
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, source in MODULE_SOURCES.items():
        target = package_dir / relative_path
        target.write_text(source)


materialize_package()
sys.path.insert(0, str(PACKAGE_ROOT))
DATA_ROOT = detect_data_root()
USE_CUDA = cuda_device_supported()

from birdclef2026.infer import main as infer_main  # noqa: E402
from birdclef2026.train import main as train_main  # noqa: E402


def run_train() -> Path:
    if USE_CUDA:
        batch_size = "{preset.gpu_batch_size}"
        max_samples = "{preset.gpu_max_samples}"
    else:
        print("Falling back to CPU training because the available GPU is not supported.")
        batch_size = "{preset.cpu_batch_size}"
        max_samples = "{preset.cpu_max_samples}"
    sys.argv = [
        "birdclef-train",
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(TRAIN_OUTPUT),
        "--architecture",
        "{preset.architecture}",
        "--backbone",
        "{preset.backbone}",
        "--epochs",
        "{preset.epochs}",
        "--batch-size",
        batch_size,
        "--num-workers",
        "4",
        "--clip-seconds",
        "5",
        "--image-height",
        "{preset.image_height}",
        "--image-width",
        "{preset.image_width}",
        "--n-mels",
        "128",
        "--transformer-dim",
        "{preset.transformer_dim}",
        "--transformer-heads",
        "{preset.transformer_heads}",
        "--transformer-layers",
        "{preset.transformer_layers}",
        "--transformer-pooling",
        "attention",
        "--min-rating",
        "0",
        "--validation-fraction",
        "0.1",
        "--max-samples",
        max_samples,
        "--loss",
        "{preset.loss}",
        "--label-smoothing",
        "{preset.label_smoothing}",
        "--mixup-alpha",
        "{preset.mixup_alpha}",
        "--random-filter-prob",
        "{preset.random_filter_prob}",
        "--freq-mask-param",
        "{preset.freq_mask_param}",
        "--time-mask-param",
        "{preset.time_mask_param}",
        "--drop-path-rate",
        "{preset.drop_path_rate}",
        "--pretrained",
    ]
    return train_main()


def run_infer(checkpoint: Path) -> Path:
    argv = [
        "birdclef-infer",
        "--data-root",
        str(DATA_ROOT),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(INFER_OUTPUT),
        "--window-seconds",
        "5",
        "--batch-size",
        "{preset.gpu_infer_batch_size}" if USE_CUDA else "{preset.cpu_infer_batch_size}",
        "--tta-shifts",
        "{preset.tta_shifts}",
    ]
    {"argv.append('--topn-postprocess')" if preset.topn_postprocess else ""}
    {"argv.append('--temporal-smoothing')" if preset.temporal_smoothing else ""}
    sys.argv = argv
    submission = infer_main()
    shutil.copy2(submission, FINAL_SUBMISSION)
    return FINAL_SUBMISSION


if __name__ == "__main__":
    checkpoint_path = run_train()
    submission_path = run_infer(checkpoint_path)
    print(f"checkpoint={{checkpoint_path}}")
    print(f"submission={{submission_path}}")
"""


def render_kernel_metadata(preset: KernelPreset) -> str:
    metadata = {
        "id": preset.kernel_id,
        "title": preset.title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": [],
        "model_sources": [],
    }
    return json.dumps(metadata, indent=2) + "\n"


def build_kernel(preset: KernelPreset) -> Path:
    kernel_dir = KAGGLE_ROOT / preset.directory_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    module_payload = build_module_payload()
    (kernel_dir / "run.py").write_text(render_run_file(module_payload, preset))
    (kernel_dir / "kernel-metadata.json").write_text(render_kernel_metadata(preset))
    return kernel_dir


def selected_presets(preset_name: str) -> list[KernelPreset]:
    if preset_name == "all":
        return [PRESETS["classifier"], PRESETS["transformer"], PRESETS["htsat"]]
    return [PRESETS[preset_name]]


def main() -> None:
    args = parse_args()
    built_dirs = [build_kernel(preset) for preset in selected_presets(args.preset)]
    for kernel_dir in built_dirs:
        print(kernel_dir)


if __name__ == "__main__":
    main()
