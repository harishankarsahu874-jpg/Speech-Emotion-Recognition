"""
model.py
--------
Defines the neural network architecture.

WHY THIS ARCHITECTURE (explain this in your presentation -- this is the
main "uniqueness" upgrade over a plain CNN or plain LSTM):

    Input (MFCC sequence: time_steps x n_mfcc)
            |
      1D Convolution layers   <- learn short, local patterns in the sound
      (like a "sliding window" that detects textures in the audio)
            |
      Bidirectional LSTM      <- learn how those patterns evolve over time,
                                  in both forward and backward direction
                                  (emotion often builds up or resolves near
                                  the end of a sentence, so reading both
                                  directions helps)
            |
      Attention layer         <- lets the model automatically learn WHICH
                                  time steps matter most for the emotion
                                  (e.g. the moment of raised pitch in anger)
                                  instead of treating every frame equally
            |
      Dense + Softmax         <- final classification into 8 emotions

A plain CNN alone loses the sense of "how sound changes over time."
A plain LSTM alone is slower to train and misses fine local texture.
CNN + BiLSTM + Attention combines the strengths of both, and the attention
layer gives you a nice bonus: you can visualize which part of speech the
model "focused on" for its decision -- great for a demo.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    """
    Simple additive attention over the time axis of an LSTM's output.
    Learns a weight for each time step, then produces a weighted sum
    (context vector) instead of just taking the last time step's output.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, time_steps, features)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)                            # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)                 # (batch, features)
        return context, weights


def build_model(input_shape, num_classes):
    """
    input_shape: (time_steps, n_mfcc), e.g. (130, 40)
    num_classes: number of emotion categories (8 for RAVDESS)
    """
    inputs = layers.Input(shape=input_shape, name="mfcc_sequence")

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    context, attn_weights = AttentionLayer(name="attention")(x)

    x = layers.Dense(128, activation="relu")(context)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SER_CNN_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model with dummy shapes and print summary.
    m = build_model(input_shape=(130, 40), num_classes=8)
    m.summary()
