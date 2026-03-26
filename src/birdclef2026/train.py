from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from birdclef2026.config import ARTIFACTS_ROOT, COMPETITION_ROOT
from birdclef2026.data import BirdClefDataset, load_examples, load_layout, split_examples
from birdclef2026.kaggle_api import DEFAULT_COMPETITION, ensure_competition_data
from birdclef2026.model import build_model


@dataclass(slots=True)
class TrainConfig:
    competition: str
    data_root: str
    output_dir: str
    architecture: str
    backbone: str
    pretrained: bool
    sample_rate: int
    clip_seconds: float
    image_height: int
    image_width: int
    n_mels: int
    n_fft: int
    hop_length: int
    fmin: int
    fmax: int
    transformer_dim: int
    transformer_heads: int
    transformer_layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    validation_fraction: float
    seed: int
    num_workers: int
    min_rating: float
    max_samples: int | None
    download_if_missing: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BirdCLEF 2026 audio classification baseline."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--data-root", type=Path, default=COMPETITION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT / "train")
    parser.add_argument(
        "--architecture",
        choices=["efficientnet_classifier", "efficientnet_transformer_sed", "htsat_token_semantic"],
        default="efficientnet_classifier",
    )
    parser.add_argument("--backbone", default="convnext_nano")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin", type=int, default=20)
    parser.add_argument("--fmax", type=int, default=16_000)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download-if-missing", action="store_true")
    return parser.parse_args()


def cuda_device_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    arch = f"sm_{major}{minor}"
    arch_list_fn = getattr(torch.cuda, "get_arch_list", None)
    if callable(arch_list_fn):
        try:
            supported_arches = arch_list_fn()
        except Exception:
            supported_arches = []
        if supported_arches:
            return arch in supported_arches
    return major >= 7


def select_device() -> torch.device:
    if cuda_device_supported():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backbone_input_size_hint(backbone: str) -> int | None:
    match = re.search(r"_(\d+)$", backbone)
    if match is None:
        return None
    return int(match.group(1))


def macro_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(targets.shape[1]):
        if targets[:, index].sum() == 0:
            continue
        scores.append(average_precision_score(targets[:, index], probabilities[:, index]))
    return float(np.mean(scores)) if scores else 0.0


def multilabel_validation_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)
    return {
        "valid_accuracy": float((predictions == targets_int).mean()),
        "valid_precision": float(
            precision_score(targets_int, predictions, average="macro", zero_division=0)
        ),
        "valid_recall": float(
            recall_score(targets_int, predictions, average="macro", zero_division=0)
        ),
        "valid_f1": float(f1_score(targets_int, predictions, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"
    losses: list[float] = []
    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        targets_all.append(targets.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(targets_all), np.concatenate(probs_all)


def main() -> Path:
    args = parse_args()
    seed_everything(args.seed)
    image_height = args.image_size or args.image_height
    image_width = args.image_size or args.image_width
    if args.architecture == "htsat_token_semantic":
        if image_height != image_width:
            raise ValueError("HTS-AT experiments require square spectrogram inputs.")
        size_hint = backbone_input_size_hint(args.backbone)
        if size_hint is not None and (image_height != size_hint or image_width != size_hint):
            raise ValueError(
                f"Backbone {args.backbone} expects {size_hint}x{size_hint} inputs, "
                f"got {image_height}x{image_width}."
            )
    if args.download_if_missing:
        ensure_competition_data(competition=args.competition, destination=args.data_root)
    layout = load_layout(args.data_root)
    examples, label_names = load_examples(
        layout,
        min_rating=args.min_rating,
        max_samples=args.max_samples,
    )
    train_examples, valid_examples = split_examples(
        examples=examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    dataset_kwargs = dict(
        label_names=label_names,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        image_height=image_height,
        image_width=image_width,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    train_dataset = BirdClefDataset(train_examples, random_crop=True, **dataset_kwargs)
    valid_dataset = BirdClefDataset(valid_examples, random_crop=False, **dataset_kwargs)
    device = select_device()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    torch.set_float32_matmul_precision("high")
    model = build_model(
        num_classes=len(label_names),
        architecture=args.architecture,
        backbone=args.backbone,
        pretrained=args.pretrained,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        competition=args.competition,
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        architecture=args.architecture,
        backbone=args.backbone,
        pretrained=args.pretrained,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        image_height=image_height,
        image_width=image_width,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        min_rating=args.min_rating,
        max_samples=args.max_samples,
        download_if_missing=args.download_if_missing,
    )

    best_score = float("-inf")
    history: list[dict[str, float]] = []
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_targets, valid_probs = run_epoch(
            model, valid_loader, criterion, None, device, None
        )
        valid_map = macro_average_precision(valid_targets, valid_probs)
        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_map": valid_map,
            **multilabel_validation_metrics(valid_targets, valid_probs),
        }
        history.append(metrics)
        print(
            " ".join(
                [
                    f"epoch={epoch}",
                    f"train_loss={train_loss:.4f}",
                    f"valid_loss={valid_loss:.4f}",
                    f"valid_map={valid_map:.4f}",
                    f"valid_accuracy={metrics['valid_accuracy']:.4f}",
                    f"valid_precision={metrics['valid_precision']:.4f}",
                    f"valid_recall={metrics['valid_recall']:.4f}",
                    f"valid_f1={metrics['valid_f1']:.4f}",
                ]
            )
        )
        if valid_map >= best_score:
            best_score = valid_map
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": label_names,
                    "config": asdict(config),
                },
                best_path,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return best_path


if __name__ == "__main__":
    checkpoint = main()
    print(checkpoint)
