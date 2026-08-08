"""
streamlit_app.py
----------------
Interactive demo app for the Speech Emotion Recognition project.

This is the piece that makes your project stand out in a demo: instead of
showing a static accuracy number in a notebook, your internship provider can
upload/record a voice clip and watch the model predict the emotion live,
with a confidence chart and a waveform view.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_artifacts, predict_emotion  # noqa: E402

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "🙂", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


@st.cache_resource
def get_model():
    return load_artifacts()


def main():
    st.title("🎙️ Speech Emotion Recognition")
    st.write(
        "Upload a short speech clip (WAV) and this CNN-BiLSTM + Attention "
        "model will predict the emotion being expressed."
    )

    try:
        model, le, mean, std = get_model()
    except Exception:
        st.warning(
            "Trained model not found yet. Run `python src/train.py` first "
            "to train and save the model, then reload this app."
        )
        return

    uploaded = st.file_uploader("Upload a .wav audio file", type=["wav"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        # Waveform visualization
        y, sr = librosa.load(tmp_path, sr=22050)
        fig, ax = plt.subplots(figsize=(8, 2.5))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#6C63FF")
        ax.set_title("Waveform")
        st.pyplot(fig)

        with st.spinner("Analyzing emotion..."):
            label, probs = predict_emotion(tmp_path, model, le, mean, std)

        emoji = EMOTION_EMOJI.get(label, "")
        st.markdown(f"## Predicted Emotion: {emoji} **{label.upper()}**")

        sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
        st.subheader("Confidence breakdown")
        st.bar_chart(sorted_probs)

        os.remove(tmp_path)

    st.markdown("---")
    st.caption(
        "Model: 1D-CNN + Bidirectional LSTM + Attention, trained on RAVDESS "
        "with noise/pitch/time-stretch augmentation."
    )


if __name__ == "__main__":
    main()
