from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from birdclef2026.audio import crop_or_pad, load_audio, waveform_to_image
from birdclef2026.config import CompetitionLayout, infer_layout


@dataclass(slots=True)
class Example:
    audio_path: Path
    labels: list[str]
    primary_label: str


def parse_secondary_labels(raw_value: object) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, list):
        return [str(value).strip() for value in raw_value if str(value).strip()]
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text in {"[]", "nan", "None"}:
        return []
    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            parsed = raw_text[1:-1].split(",")
        return [str(value).strip().strip("'\"") for value in parsed if str(value).strip()]
    return [part.strip() for part in raw_text.replace("|", ",").split(",") if part.strip()]


def load_layout(root: Path | None = None) -> CompetitionLayout:
    layout = infer_layout(root)
    required = [layout.train_csv, layout.train_audio, layout.sample_submission]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Competition files are missing: {missing_str}")
    return layout


def resolve_audio_path(row: pd.Series, train_audio_dir: Path) -> Path:
    candidate_columns = ["filepath", "path"]
    for column in candidate_columns:
        if column in row and isinstance(row[column], str) and row[column].strip():
            path = Path(row[column])
            if path.is_file():
                return path
            joined = train_audio_dir / row[column]
            if joined.is_file():
                return joined
    filename = str(row.get("filename", "")).strip()
    if not filename:
        raise ValueError("The metadata row does not contain a usable filename.")
    nested = train_audio_dir / str(row.get("primary_label", "")).strip() / filename
    if nested.is_file():
        return nested
    flat = train_audio_dir / filename
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"Audio file not found for row: {filename}")


def build_label_vocabulary(metadata: pd.DataFrame, sample_submission: pd.DataFrame) -> list[str]:
    submission_labels = [column for column in sample_submission.columns if column != "row_id"]
    if submission_labels:
        return submission_labels
    labels: set[str] = set(metadata["primary_label"].dropna().astype(str))
    if "secondary_labels" in metadata:
        for raw_value in metadata["secondary_labels"]:
            labels.update(parse_secondary_labels(raw_value))
    return sorted(labels)


def load_examples(
    layout: CompetitionLayout,
    min_rating: float = 0.0,
    max_samples: int | None = None,
) -> tuple[list[Example], list[str]]:
    metadata = pd.read_csv(layout.train_csv)
    sample_submission = pd.read_csv(layout.sample_submission)
    label_names = build_label_vocabulary(metadata, sample_submission)
    label_set = set(label_names)
    examples: list[Example] = []
    for _, row in metadata.iterrows():
        rating = float(row.get("rating", 0.0))
        if rating < min_rating:
            continue
        try:
            audio_path = resolve_audio_path(row, layout.train_audio)
        except FileNotFoundError:
            continue
        primary_label = str(row["primary_label"]).strip()
        labels = [primary_label]
        if "secondary_labels" in row:
            labels.extend(parse_secondary_labels(row["secondary_labels"]))
        labels = [label for label in labels if label in label_set]
        if not labels:
            continue
        examples.append(
            Example(
                audio_path=audio_path,
                labels=sorted(set(labels)),
                primary_label=primary_label,
            )
        )
        if max_samples and len(examples) >= max_samples:
            break
    if not examples:
        raise RuntimeError("No training examples were found in the competition files.")
    return examples, label_names


def split_examples(
    examples: list[Example],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.primary_label].append(example)
    train_examples: list[Example] = []
    valid_examples: list[Example] = []
    rng = np.random.default_rng(seed)
    for _label, group in by_label.items():
        if len(group) == 1:
            train_examples.extend(group)
            continue
        order = np.arange(len(group))
        rng.shuffle(order)
        split_idx = max(1, int(len(group) * (1 - validation_fraction)))
        split_idx = min(split_idx, len(group) - 1)
        for idx in order[:split_idx]:
            train_examples.append(group[idx])
        for idx in order[split_idx:]:
            valid_examples.append(group[idx])
    if not valid_examples:
        valid_examples = train_examples[-max(1, len(train_examples) // 10) :]
        train_examples = train_examples[: -len(valid_examples)]
    return train_examples, valid_examples


class BirdClefDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples: Iterable[Example],
        label_names: list[str],
        sample_rate: int,
        clip_seconds: float,
        image_height: int,
        image_width: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        fmin: int,
        fmax: int,
        random_crop: bool,
    ) -> None:
        self.examples = list(examples)
        self.label_names = label_names
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.image_height = image_height
        self.image_width = image_width
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.random_crop = random_crop
        self.label_to_index = {label: index for index, label in enumerate(label_names)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        waveform = load_audio(example.audio_path, self.sample_rate)
        waveform = crop_or_pad(
            waveform=waveform,
            sample_rate=self.sample_rate,
            clip_seconds=self.clip_seconds,
            random_crop=self.random_crop,
        )
        image = waveform_to_image(
            waveform=waveform,
            sample_rate=self.sample_rate,
            image_height=self.image_height,
            image_width=self.image_width,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        target = torch.zeros(len(self.label_names), dtype=torch.float32)
        for label in example.labels:
            target[self.label_to_index[label]] = 1.0
        return image, target


def build_submission_map(sample_submission: pd.DataFrame) -> dict[str, list[tuple[int, str]]]:
    mapping: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if {"filename", "seconds", "row_id"}.issubset(sample_submission.columns):
        for row in sample_submission.itertuples(index=False):
            mapping[Path(row.filename).stem].append((int(row.seconds), str(row.row_id)))
        return dict(mapping)
    for row_id in sample_submission["row_id"].astype(str):
        stem, seconds = parse_row_id(row_id)
        mapping[stem].append((seconds, row_id))
    return dict(mapping)


def parse_row_id(row_id: str) -> tuple[str, int]:
    stem, _, tail = row_id.rpartition("_")
    if not stem or not tail.isdigit():
        raise ValueError(f"Unrecognized row_id format: {row_id}")
    return stem, int(tail)
