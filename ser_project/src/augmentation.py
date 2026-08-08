"""
augmentation.py
----------------
Audio data augmentation for Speech Emotion Recognition.

WHY THIS MATTERS (explain this in your report/demo):
Emotion datasets like RAVDESS are small (~1440 clips). A model trained only on
the original clips tends to memorize speaker-specific traits instead of
emotion-specific traits. Augmentation creates realistic variations of the same
clip so the model learns to generalize across noise conditions, pitch, and
speaking speed -- this is one of the main "uniqueness" additions of this
project compared to a typical student submission.

Each function takes a 1D numpy waveform and returns a modified waveform of
the same approximate length.
"""

import numpy as np
import librosa


def add_noise(y, noise_factor=0.005):
    """Add light Gaussian noise to simulate real-world recording conditions."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def pitch_shift(y, sr, n_steps=2):
    """Shift pitch up/down by n_steps semitones (simulates different vocal ranges)."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate=1.1):
    """Speed up (rate > 1) or slow down (rate < 1) the speech without changing pitch."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def shift_time(y, sr, shift_max=0.2):
    """Shift the waveform left/right in time (simulates timing variation in recording)."""
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(y, shift)


def augment_sample(y, sr):
    """
    Returns a dict of {augmentation_name: augmented_waveform} for one input clip.
    Used during training-set preparation to multiply the amount of training data.
    """
    augmented = {
        "original": y,
        "noise": add_noise(y),
        "pitch_up": pitch_shift(y, sr, n_steps=2),
        "pitch_down": pitch_shift(y, sr, n_steps=-2),
        "stretch": time_stretch(y, rate=0.9),
        "time_shift": shift_time(y, sr),
    }
    return augmented
