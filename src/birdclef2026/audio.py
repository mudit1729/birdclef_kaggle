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


def random_filtering(
    waveform: np.ndarray,
    sample_rate: int,
    prob: float = 0.5,
) -> np.ndarray:
    """Apply random biquad peaking EQ filter to simulate microphone variation."""
    if random.random() > prob:
        return waveform
    from scipy.signal import lfilter

    center_freq = random.uniform(200, 8000)
    gain_db = random.uniform(-6, 6)
    q_factor = random.uniform(0.5, 2.0)
    w0 = 2.0 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2.0 * q_factor)
    a_lin = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a_lin
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, waveform).astype(np.float32)


def spec_augment(
    image: torch.Tensor,
    freq_mask_param: int = 0,
    time_mask_param: int = 0,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> torch.Tensor:
    """Apply SpecAugment-style time and frequency masking to a spectrogram image.

    Args:
        image: (C, H, W) tensor where H=frequency, W=time.
        freq_mask_param: Maximum width of frequency masks (0 to disable).
        time_mask_param: Maximum width of time masks (0 to disable).
    """
    if freq_mask_param <= 0 and time_mask_param <= 0:
        return image
    image = image.clone()
    _, h, w = image.shape
    for _ in range(num_freq_masks):
        if freq_mask_param > 0 and h > 1:
            f = random.randint(0, min(freq_mask_param, h - 1))
            f0 = random.randint(0, h - f)
            image[:, f0 : f0 + f, :] = 0.0
    for _ in range(num_time_masks):
        if time_mask_param > 0 and w > 1:
            t = random.randint(0, min(time_mask_param, w - 1))
            t0 = random.randint(0, w - t)
            image[:, :, t0 : t0 + t] = 0.0
    return image


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
