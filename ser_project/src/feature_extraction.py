"""
feature_extraction.py
----------------------
Converts a raw audio waveform into a fixed-size numeric feature vector that a
neural network can learn from.

CONCEPT RECAP (for your report):
Raw audio is just a long list of amplitude values (e.g. 3 seconds at 22050 Hz
= ~66,000 numbers). That's too long and too "raw" for a small model to learn
emotion from directly. Instead we extract features that describe the SHAPE of
the sound:

- MFCCs (Mel-Frequency Cepstral Coefficients): describe the timbre/tone
  color of the voice -- the standard feature for speech tasks.
- Chroma: describes pitch-class energy, useful for tone/intonation shifts
  that carry emotional meaning (e.g. rising pitch in surprise).
- Mel Spectrogram: describes energy across frequency bands over time.
- Zero Crossing Rate (ZCR): how "noisy"/harsh a sound is -- angry or fearful
  speech tends to have a higher ZCR than calm speech.
- Root Mean Square Energy (RMS): loudness over time -- angry/happy speech is
  typically louder than sad/calm speech.

Combining MFCC + Chroma + Mel + ZCR + RMS (instead of MFCC alone) is the
second "uniqueness" upgrade in this project -- it gives the model both
tonal and energy-based cues instead of relying on timbre alone.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 3.0          # seconds -- clips are padded/truncated to this length
N_MFCC = 40


def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    """Load an audio file and force it to a fixed duration (pad or trim)."""
    y, sr = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def extract_features(y, sr=SAMPLE_RATE):
    """
    Extract and combine multiple feature types from a waveform.
    Returns a 1D numpy array (fixed length) representing the clip.
    Each feature is averaged over time (mean) so the final vector length
    doesn't depend on clip duration.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr.T, axis=0)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)

    features = np.concatenate([mfcc_mean, chroma_mean, mel_mean, zcr_mean, rms_mean])
    return features


def extract_features_sequence(y, sr=SAMPLE_RATE, max_len=130):
    """
    Extract MFCC as a TIME SEQUENCE (not averaged) for use with CNN-LSTM models,
    which need a (time_steps, features) shape instead of a single flat vector.
    Pads/truncates the time axis to max_len frames so all samples have the
    same shape.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T  # (time, n_mfcc)

    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc  # shape: (max_len, N_MFCC)
