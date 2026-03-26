from __future__ import annotations

import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def crop_or_pad(
    waveform: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    random_crop: bool,
) -> np.ndarray:
    target_length = int(sample_rate * clip_seconds)
    if waveform.shape[0] == target_length:
        return waveform
    if waveform.shape[0] > target_length:
        max_offset = waveform.shape[0] - target_length
        offset = random.randint(0, max_offset) if random_crop and max_offset > 0 else 0
        return waveform[offset : offset + target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: waveform.shape[0]] = waveform
    return padded


def waveform_to_image(
    waveform: np.ndarray,
    sample_rate: int,
    image_height: int,
    image_width: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: int,
    fmax: int,
) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    image = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    image = F.interpolate(
        image,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return image.squeeze(0)
