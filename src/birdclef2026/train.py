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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from birdclef2026.config import ARTIFACTS_ROOT, COMPETITION_ROOT
from birdclef2026.data import BirdClefDataset, Example, load_examples, load_layout, split_examples
from birdclef2026.kaggle_api import DEFAULT_COMPETITION, ensure_competition_data
from birdclef2026.model import build_model, combine_primary_secondary_probabilities


@dataclass(slots=True)
class TrainConfig:
    competition: str
    data_root: str
    output_dir: str
    architecture: str
    backbone: str
    classifier_head_mode: str
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
    transformer_pooling: str
    dropout: float
    drop_path_rate: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    primary_loss_weight: float
    secondary_loss_weight: float
    loss_name: str
    focal_gamma: float
    focal_alpha_min: float
    focal_alpha_max: float
    label_smoothing: float
    mixup_alpha: float
    random_filter_prob: float
    freq_mask_param: int
    time_mask_param: int
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
    parser.add_argument(
        "--classifier-head-mode",
        choices=["single", "dual"],
        default="dual",
    )
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
    parser.add_argument(
        "--transformer-pooling",
        choices=["clip_attention", "attention", "mean", "max"],
        default="attention",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--primary-loss-weight", type=float, default=2.0)
    parser.add_argument("--secondary-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss", choices=["bce", "focal", "soft_auc"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-min", type=float, default=0.05)
    parser.add_argument("--focal-alpha-max", type=float, default=0.95)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--random-filter-prob", type=float, default=0.0)
    parser.add_argument("--freq-mask-param", type=int, default=0)
    parser.add_argument("--time-mask-param", type=int, default=0)
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
    thresholds: np.ndarray,
) -> dict[str, float]:
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int32)
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
        "valid_precision_micro": float(
            precision_score(targets_int, predictions, average="micro", zero_division=0)
        ),
        "valid_recall_micro": float(
            recall_score(targets_int, predictions, average="micro", zero_division=0)
        ),
        "valid_f1_micro": float(
            f1_score(targets_int, predictions, average="micro", zero_division=0)
        ),
    }


def compute_label_counts(
    examples: list[Example],
    label_names: list[str],
    mode: str,
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    counts = np.zeros(len(label_names), dtype=np.float32)
    for example in examples:
        if mode == "primary":
            counts[label_to_index[example.primary_label]] += 1.0
            continue
        if mode == "secondary":
            labels = [label for label in example.labels if label != example.primary_label]
        elif mode == "combined":
            labels = example.labels
        else:
            raise ValueError(f"Unknown count mode: {mode}")
        for label in set(labels):
            index = label_to_index.get(label)
            if index is not None:
                counts[index] += 1.0
    return counts


def normalized_inverse_frequency_weights(counts: np.ndarray) -> np.ndarray:
    weights = 1.0 / (counts.astype(np.float32) + 1.0)
    mean = float(weights.mean()) if weights.size else 1.0
    if mean <= 0.0:
        return np.ones_like(weights, dtype=np.float32)
    return (weights / mean).astype(np.float32)


def compute_class_focal_alpha(
    counts: np.ndarray,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    if not 0.0 < alpha_min < 1.0:
        raise ValueError(f"Expected focal alpha min in (0, 1), got {alpha_min}.")
    if not 0.0 < alpha_max < 1.0:
        raise ValueError(f"Expected focal alpha max in (0, 1), got {alpha_max}.")
    if alpha_min > alpha_max:
        raise ValueError(
            f"Expected focal alpha min <= max, got min={alpha_min} max={alpha_max}."
        )
    inv_weights = normalized_inverse_frequency_weights(counts)
    inv_min = float(inv_weights.min()) if inv_weights.size else 0.0
    inv_max = float(inv_weights.max()) if inv_weights.size else 0.0
    if inv_max - inv_min < 1e-12:
        midpoint = (alpha_min + alpha_max) / 2.0
        return np.full_like(inv_weights, midpoint, dtype=np.float32)
    scaled = (inv_weights - inv_min) / (inv_max - inv_min)
    alpha = alpha_min + scaled * (alpha_max - alpha_min)
    return alpha.astype(np.float32)


class MultilabelFocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * bce
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_factor * loss
        return loss.mean()


class SoftAUCLoss(nn.Module):
    """Differentiable approximation of ROC-AUC, directly optimizing the competition metric.

    Computes pairwise log-sigmoid loss between positive and negative samples per class.
    Falls back to BCE for classes with no positive/negative samples in the batch.
    """

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        auc_losses: list[torch.Tensor] = []
        for c in range(targets.shape[1]):
            pos_mask = targets[:, c] > 0.5
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            pos_probs = probs[pos_mask, c]
            neg_probs = probs[neg_mask, c]
            diff = pos_probs.unsqueeze(1) - neg_probs.unsqueeze(0)
            auc_losses.append(-torch.log(torch.sigmoid(diff) + 1e-7).mean())
        if not auc_losses:
            return bce
        auc_loss = torch.stack(auc_losses).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * auc_loss


class MultilabelCriterion(nn.Module):
    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self,
        outputs: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.base_loss(outputs, targets["combined"])


class DualHeadClassifierCriterion(nn.Module):
    def __init__(
        self,
        primary_class_weights: torch.Tensor,
        secondary_criterion: nn.Module,
        primary_loss_weight: float,
        secondary_loss_weight: float,
    ) -> None:
        super().__init__()
        self.register_buffer("primary_class_weights", primary_class_weights)
        self.secondary_criterion = secondary_criterion
        self.primary_loss_weight = primary_loss_weight
        self.secondary_loss_weight = secondary_loss_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        primary_loss = F.cross_entropy(
            outputs["primary_logits"],
            targets["primary_index"],
            weight=self.primary_class_weights.to(
                device=outputs["primary_logits"].device,
                dtype=outputs["primary_logits"].dtype,
            ),
        )
        secondary_loss = self.secondary_criterion(
            outputs["secondary_logits"],
            targets["secondary"],
        )
        total_weight = self.primary_loss_weight + self.secondary_loss_weight
        return (
            (self.primary_loss_weight * primary_loss)
            + (self.secondary_loss_weight * secondary_loss)
        ) / total_weight


def output_probabilities(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, dict):
        return combine_primary_secondary_probabilities(
            outputs["primary_logits"],
            outputs["secondary_logits"],
        )
    return torch.sigmoid(outputs)


def sweep_per_class_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
    sweep: list[dict[str, float]] = []
    for index in range(targets.shape[1]):
        class_targets = targets[:, index].astype(np.int32)
        if class_targets.sum() == 0:
            sweep.append(
                {
                    "class_index": float(index),
                    "best_threshold": 0.5,
                    "best_f1": 0.0,
                    "positive_count": float(class_targets.sum()),
                }
            )
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in threshold_grid:
            predictions = (probabilities[:, index] >= threshold).astype(np.int32)
            score = f1_score(class_targets, predictions, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[index] = best_threshold
        sweep.append(
            {
                "class_index": float(index),
                "best_threshold": best_threshold,
                "best_f1": best_f1,
                "positive_count": float(class_targets.sum()),
            }
        )
    return thresholds, sweep


def apply_label_smoothing(
    targets: dict[str, torch.Tensor],
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """Apply label smoothing to multilabel targets: 0->eps, 1->1-eps."""
    if label_smoothing <= 0.0:
        return targets
    smoothed = dict(targets)
    for key in ("combined", "secondary"):
        if key in smoothed:
            t = smoothed[key]
            smoothed[key] = t * (1.0 - label_smoothing) + (1.0 - t) * label_smoothing
    return smoothed


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    label_smoothing: float = 0.0,
    scheduler: object | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"
    losses: list[float] = []
    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        raw_targets = {
            name: value.to(device)
            for name, value in targets.items()
        }
        smooth_targets = (
            apply_label_smoothing(raw_targets, label_smoothing) if training else raw_targets
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, smooth_targets)
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
        targets_all.append(raw_targets["combined"].detach().cpu().numpy())
        probs_all.append(output_probabilities(outputs).detach().cpu().numpy())

    if training and scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step()

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
    train_dataset = BirdClefDataset(
        train_examples,
        random_crop=True,
        mixup_alpha=args.mixup_alpha,
        random_filter_prob=args.random_filter_prob,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        **dataset_kwargs,
    )
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
        classifier_head_mode=args.classifier_head_mode,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_pooling=args.transformer_pooling,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    combined_counts = compute_label_counts(train_examples, label_names, mode="combined")
    primary_counts = compute_label_counts(train_examples, label_names, mode="primary")
    secondary_counts = compute_label_counts(train_examples, label_names, mode="secondary")
    combined_focal_alpha = compute_class_focal_alpha(
        combined_counts,
        alpha_min=args.focal_alpha_min,
        alpha_max=args.focal_alpha_max,
    )
    secondary_focal_alpha = compute_class_focal_alpha(
        secondary_counts,
        alpha_min=args.focal_alpha_min,
        alpha_max=args.focal_alpha_max,
    )
    primary_class_weights = normalized_inverse_frequency_weights(primary_counts)
    if args.loss == "soft_auc":
        single_head_loss = SoftAUCLoss(bce_weight=0.5)
        secondary_loss = SoftAUCLoss(bce_weight=0.5)
    elif args.loss == "focal":
        single_head_loss = MultilabelFocalLoss(
            gamma=args.focal_gamma,
            alpha=torch.tensor(combined_focal_alpha, dtype=torch.float32),
        )
        secondary_loss = MultilabelFocalLoss(
            gamma=args.focal_gamma,
            alpha=torch.tensor(secondary_focal_alpha, dtype=torch.float32),
        )
    else:
        single_head_loss = nn.BCEWithLogitsLoss()
        secondary_loss = nn.BCEWithLogitsLoss()
    if args.architecture == "efficientnet_classifier" and args.classifier_head_mode == "dual":
        criterion = DualHeadClassifierCriterion(
            primary_class_weights=torch.tensor(primary_class_weights, dtype=torch.float32),
            secondary_criterion=secondary_loss,
            primary_loss_weight=args.primary_loss_weight,
            secondary_loss_weight=args.secondary_loss_weight,
        ).to(device)
    else:
        criterion = MultilabelCriterion(single_head_loss).to(device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        competition=args.competition,
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        architecture=args.architecture,
        backbone=args.backbone,
        classifier_head_mode=args.classifier_head_mode,
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
        transformer_pooling=args.transformer_pooling,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        primary_loss_weight=args.primary_loss_weight,
        secondary_loss_weight=args.secondary_loss_weight,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha_min=args.focal_alpha_min,
        focal_alpha_max=args.focal_alpha_max,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        random_filter_prob=args.random_filter_prob,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        min_rating=args.min_rating,
        max_samples=args.max_samples,
        download_if_missing=args.download_if_missing,
    )
    weighting_payload = [
        {
            "label": label,
            "primary_count": int(primary_count),
            "secondary_count": int(secondary_count),
            "combined_count": int(combined_count),
            "primary_weight": float(primary_weight),
            "secondary_alpha": float(secondary_alpha),
            "combined_alpha": float(combined_alpha),
        }
        for (
            label, primary_count, secondary_count, combined_count,
            primary_weight, secondary_alpha, combined_alpha,
        ) in zip(
            label_names,
            primary_counts,
            secondary_counts,
            combined_counts,
            primary_class_weights,
            secondary_focal_alpha,
            combined_focal_alpha,
            strict=True,
        )
    ]
    (output_dir / "class_alpha.json").write_text(json.dumps(weighting_payload, indent=2))
    print(
        " ".join(
            [
                f"loss={args.loss}",
                f"focal_gamma={args.focal_gamma:.2f}",
                f"combined_alpha_min={float(combined_focal_alpha.min()):.4f}",
                f"combined_alpha_max={float(combined_focal_alpha.max()):.4f}",
                f"secondary_alpha_min={float(secondary_focal_alpha.min()):.4f}",
                f"secondary_alpha_max={float(secondary_focal_alpha.max()):.4f}",
            ]
        )
    )

    best_score = float("-inf")
    history: list[dict[str, float]] = []
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            label_smoothing=args.label_smoothing, scheduler=scheduler,
        )
        valid_loss, valid_targets, valid_probs = run_epoch(
            model, valid_loader, criterion, None, device, None
        )
        valid_map = macro_average_precision(valid_targets, valid_probs)
        tuned_thresholds, threshold_sweep = sweep_per_class_thresholds(valid_targets, valid_probs)
        tuned_metrics = multilabel_validation_metrics(valid_targets, valid_probs, tuned_thresholds)
        fixed_metrics = {
            f"{name}_fixed050": value
            for name, value in multilabel_validation_metrics(
                valid_targets,
                valid_probs,
                np.full(valid_targets.shape[1], 0.5, dtype=np.float32),
            ).items()
        }
        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_map": valid_map,
            "valid_threshold_min": float(tuned_thresholds.min()),
            "valid_threshold_max": float(tuned_thresholds.max()),
            "valid_threshold_mean": float(tuned_thresholds.mean()),
            **tuned_metrics,
            **fixed_metrics,
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
                    f"valid_precision_macro={metrics['valid_precision']:.4f}",
                    f"valid_recall_macro={metrics['valid_recall']:.4f}",
                    f"valid_f1_macro={metrics['valid_f1']:.4f}",
                    f"valid_precision_micro={metrics['valid_precision_micro']:.4f}",
                    f"valid_recall_micro={metrics['valid_recall_micro']:.4f}",
                    f"valid_f1_micro={metrics['valid_f1_micro']:.4f}",
                ]
            )
        )
        if valid_map >= best_score:
            best_score = valid_map
            (output_dir / "validation_thresholds.json").write_text(
                json.dumps(
                    {
                        "thresholds": [
                            {
                                "label": label,
                                "best_threshold": float(threshold),
                                "best_f1": float(sweep_item["best_f1"]),
                                "positive_count": int(sweep_item["positive_count"]),
                            }
                            for label, threshold, sweep_item in zip(
                                label_names,
                                tuned_thresholds,
                                threshold_sweep,
                                strict=True,
                            )
                        ]
                    },
                    indent=2,
                )
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": label_names,
                    "config": asdict(config),
                    "validation_thresholds": tuned_thresholds.tolist(),
                },
                best_path,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return best_path


if __name__ == "__main__":
    checkpoint = main()
    print(checkpoint)
