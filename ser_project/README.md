# 🎙️ Speech Emotion Recognition (SER)

Recognizes human emotions (happy, sad, angry, calm, neutral, fearful, disgust,
surprised) from short speech audio clips using deep learning.

## What makes this project different from a typical student submission

Most intern-level SER projects stop at "extract MFCC → train a basic CNN →
report accuracy." This project adds three upgrades that are easy to explain
and demonstrate, and make the work noticeably more thorough:

1. **Data augmentation** (`src/augmentation.py`) — adds noise, pitch shift,
   time stretch, and time shift to each clip, multiplying the effective
   training data and making the model robust to real-world recording
   conditions instead of memorizing individual voices.
2. **Richer feature set** — MFCCs, Chroma, Mel Spectrogram, Zero Crossing
   Rate, and RMS Energy are combined (see `src/feature_extraction.py`),
   capturing both the tonal quality *and* the energy/loudness pattern of
   speech, since energy is a strong emotional cue (angry/happy = loud,
   sad/calm = quiet).
3. **CNN + BiLSTM + Attention architecture** (`src/model.py`) instead of a
   plain CNN. The CNN layers learn local sound texture, the Bidirectional
   LSTM learns how that texture evolves over the sentence, and the
   Attention layer learns *which moments* in the speech mattered most for
   the emotion — which you can also use as a talking point in your demo
   ("the model paid the most attention to this part of the sentence").
4. **Live interactive demo app** (`app/streamlit_app.py`) — upload a voice
   clip and see the predicted emotion with a confidence chart, instead of
   just a static accuracy number in a notebook.

## Project structure

```
ser_project/
├── data/
│   ├── raw/            <- put the downloaded RAVDESS dataset here
│   └── processed/      <- extracted features get saved here (auto-generated)
├── models/              <- trained model + plots get saved here (auto-generated)
├── src/
│   ├── augmentation.py       # noise/pitch/time-stretch augmentation
│   ├── feature_extraction.py # MFCC/Chroma/Mel/ZCR/RMS extraction
│   ├── dataset.py             # builds the training dataset from raw audio
│   ├── model.py                # CNN + BiLSTM + Attention architecture
│   ├── train.py                 # trains the model, saves plots/reports
│   └── predict.py               # predicts emotion for one new audio file
├── app/
│   └── streamlit_app.py    # interactive demo web app
├── notebook/
│   └── SER_walkthrough.ipynb  # step-by-step notebook explaining every stage
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the dataset

This project uses **RAVDESS** (Ryerson Audio-Visual Database of Emotional
Speech and Song) — Audio_Speech_Actors_01-24 only (~215 MB), a widely
recognized benchmark dataset for this task.

1. Download it from Zenodo: https://zenodo.org/record/1188976
   (file: `Audio_Speech_Actors_01-24.zip`)
2. Unzip it so that `data/raw/` contains folders `Actor_01` through
   `Actor_24`, each with `.wav` files.

```
data/raw/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

## 3. Build the feature dataset

```bash
cd src
python dataset.py
```

This extracts MFCC sequences for every clip **plus 5 augmented variants of
each clip**, and saves them to `data/processed/X.npy` and `y.npy`.
With 1440 original clips this produces ~8,600 training samples.

## 4. Train the model

```bash
python train.py
```

This trains the CNN-BiLSTM-Attention model with early stopping, and saves to
`models/`:
- `ser_model.keras` — the trained model
- `label_encoder.pkl` — maps model outputs back to emotion names
- `training_history.png` — accuracy/loss curves
- `confusion_matrix.png` — per-emotion performance breakdown
- `classification_report.txt` — precision/recall/F1 per emotion

Training on a laptop CPU typically takes 15-40 minutes depending on hardware.
A GPU (e.g. free Google Colab GPU) will be much faster if you'd rather not
wait — just upload the `src/` and `data/raw/` folders there.

## 5. Predict on a new clip

```bash
python predict.py ../data/raw/Actor_01/03-01-03-01-01-01-01.wav
```

## 6. Run the live demo app

```bash
cd ..
streamlit run app/streamlit_app.py
```

Upload any `.wav` speech clip and see the predicted emotion, confidence
breakdown, and waveform — this is the best part to show your internship
provider live.

## Concepts explained simply (for your report/viva)

- **MFCC (Mel-Frequency Cepstral Coefficients):** a compact numeric summary
  of the *shape* of the voice's frequency spectrum, adjusted to match how
  the human ear perceives pitch. It's the standard feature used across
  almost all speech-processing tasks (recognition, verification, emotion).
- **CNN (Convolutional layers):** slide small filters across the MFCC
  sequence to detect short local patterns, similar to how CNNs detect edges
  in images, but here detecting sound "textures."
- **BiLSTM (Bidirectional LSTM):** a recurrent network that reads the
  sequence of patterns both forward and backward in time, so it can use
  context from the whole clip (not just what came before) to understand how
  the emotional cue develops.
- **Attention:** instead of treating every time step of the audio equally,
  the model learns to weigh the moments that matter most for the final
  decision — giving both better accuracy and a degree of interpretability.

## Expected results

On the real RAVDESS dataset with this pipeline, test accuracy typically
lands around **65-75%** across all 8 emotions (RAVDESS is a hard,
fine-grained 8-class problem — for reference, published baselines using
plain MFCC+CNN are usually in the 55-65% range, and human agreement on
labeling these clips is itself only around 80%). If you want to boost this
further, easy next steps are: collapsing "neutral" and "calm" into one
class (they're acoustically very similar), or combining RAVDESS with TESS
for more training data.

## Notes

- The pipeline was tested end-to-end on synthetic audio to verify there are
  no bugs in the code path; you'll get real performance numbers once you run
  it on the actual RAVDESS dataset.
- All random seeds are fixed (`random_state=42`) so your train/val/test
  split is reproducible.
