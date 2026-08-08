"""
train.py
--------
Loads the processed feature dataset (X.npy, y.npy), splits into
train/validation/test, trains the CNN-BiLSTM-Attention model, and saves:
    - the trained model      -> models/ser_model.keras
    - the label encoder      -> models/label_encoder.pkl
    - training history plot  -> models/training_history.png
    - a classification report + confusion matrix on the held-out test set

Run this AFTER running dataset.py to generate the processed features.
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data():
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    return X, y


def main():
    print("Loading processed data...")
    X, y = load_processed_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Normalize features (zero mean, unit variance) -- helps the network train faster
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-8)
    np.save(os.path.join(MODELS_DIR, "norm_stats.npy"), np.array([mean, std]))

    # Encode string labels ("happy", "sad", ...) into integers, then one-hot
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    y_onehot = to_categorical(y_int)

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Split: 70% train, 15% validation, 15% test (stratified so each emotion
    # is represented proportionally in every split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=0.30, random_state=42, stratify=y_int
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    model = build_model(input_shape=X.shape[1:], num_classes=y_onehot.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "ser_model.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "training_history.png"), dpi=150)
    print("Saved training_history.png")

    # Evaluate on held-out test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=le.classes_)
    print("\nClassification Report:\n", report)
    with open(os.path.join(MODELS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Speech Emotion Recognition")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=150)
    print("Saved confusion_matrix.png")

    print("\nDone. Trained model saved at:", os.path.join(MODELS_DIR, "ser_model.keras"))


if __name__ == "__main__":
    main()
