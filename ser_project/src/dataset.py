"""
dataset.py
----------
Builds the training dataset from raw RAVDESS audio files.

RAVDESS FILENAME FORMAT (this is how labels are obtained -- no manual
labeling needed):
    03-01-06-01-02-01-12.wav
    |  |  |  |  |  |  |
    |  |  |  |  |  |  actor number (odd = male, even = female)
    |  |  |  |  |  repetition (1st or 2nd take)
    |  |  |  |  statement (1st or 2nd sentence)
    |  |  |  emotion intensity (1 = normal, 2 = strong)
    |  |  emotion  <-- this is the label we care about
    |  vocal channel (01 = speech, 02 = song)
    modality (01 = full AV, 02 = video only, 03 = audio only)

We only use audio-only speech files: modality 03, vocal channel 01.

Emotion code -> label mapping:
    01 neutral, 02 calm, 03 happy, 04 sad,
    05 angry, 06 fearful, 07 disgust, 08 surprised
"""

import os
import glob
import numpy as np
from tqdm import tqdm

from feature_extraction import load_audio, extract_features_sequence, SAMPLE_RATE
from augmentation import augment_sample

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

LABELS = list(EMOTION_MAP.values())


def parse_label(filename):
    """Extract the emotion label from a RAVDESS filename."""
    parts = os.path.basename(filename).split("-")
    emotion_code = parts[2]
    return EMOTION_MAP[emotion_code]


def build_dataset(raw_dir, out_dir, use_augmentation=True):
    """
    Walk through all .wav files in raw_dir (RAVDESS Actor_* folders),
    extract MFCC sequences (+ augmented versions), and save:
        X.npy  -> shape (num_samples, time_steps, n_mfcc)
        y.npy  -> shape (num_samples,) string labels
    """
    files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    # Keep only audio-only speech files (modality 03, vocal channel 01)
    files = [f for f in files if os.path.basename(f).startswith("03-01")]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No RAVDESS .wav files found under {raw_dir}. "
            "Download the dataset and place Actor_01..Actor_24 folders there. "
            "See README.md for the download link."
        )

    X, y = [], []

    for f in tqdm(files, desc="Extracting features"):
        label = parse_label(f)
        waveform, sr = load_audio(f, sr=SAMPLE_RATE)

        if use_augmentation:
            variants = augment_sample(waveform, sr)
        else:
            variants = {"original": waveform}

        for variant_wave in variants.values():
            seq = extract_features_sequence(variant_wave, sr=sr)
            X.append(seq)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    print(f"Saved {X.shape[0]} samples with shape {X.shape[1:]} to {out_dir}")
    return X, y


if __name__ == "__main__":
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    build_dataset(RAW_DIR, OUT_DIR, use_augmentation=True)
