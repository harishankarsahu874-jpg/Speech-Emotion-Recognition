"""
predict.py
----------
Loads the trained model and predicts the emotion for a single new audio file.
Used by both the command line and the Streamlit demo app.
"""

import os
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from feature_extraction import load_audio, extract_features_sequence
from model import AttentionLayer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model = load_model(
        os.path.join(MODELS_DIR, "ser_model.keras"),
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    mean, std = np.load(os.path.join(MODELS_DIR, "norm_stats.npy"))
    return model, le, mean, std


def predict_emotion(audio_path, model=None, le=None, mean=None, std=None):
    """
    Predicts the emotion for a single audio file.
    Returns: (predicted_label, dict of {label: probability})
    """
    if model is None:
        model, le, mean, std = load_artifacts()

    y, sr = load_audio(audio_path)
    features = extract_features_sequence(y, sr=sr)
    features = (features - mean) / (std + 1e-8)
    features = np.expand_dims(features, axis=0)  # add batch dimension

    probs = model.predict(features, verbose=0)[0]
    predicted_label = le.classes_[np.argmax(probs)]
    prob_dict = {label: float(p) for label, p in zip(le.classes_, probs)}

    return predicted_label, prob_dict


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_wav_file>")
        sys.exit(1)

    label, probs = predict_emotion(sys.argv[1])
    print(f"\nPredicted emotion: {label.upper()}\n")
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {emo:12s}: {p*100:.2f}%")
