"""
CaptionIQ — Attention-Based CNN-LSTM Caption Generation Model
Uses Bahdanau (additive) attention over spatial CNN features for
image-specific caption generation.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, Concatenate, Layer
)
from tensorflow.keras.optimizers import Adam

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    EMBED_DIM, LSTM_UNITS, DROPOUT_RATE, FEATURE_DIM,
    FEATURE_LOCATIONS, ATTENTION_DIM, LEARNING_RATE,
)


class BahdanauAttention(Layer):
    """
    Bahdanau (additive) attention over spatial image features.

    Given spatial features (batch, 49, 512) and LSTM hidden state (batch, 512):
        score = V * tanh(W1 * features + W2 * hidden)
        weights = softmax(score)
        context = sum(weights * features)

    This lets the model focus on different image regions for each word.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units, name="att_features")
        self.W2 = Dense(units, name="att_hidden")
        self.V = Dense(1, name="att_score")

    def call(self, inputs):
        features, hidden = inputs
        # features: (batch, locations, feature_dim)
        # hidden:   (batch, lstm_units)

        hidden_expanded = tf.expand_dims(hidden, 1)  # (batch, 1, lstm_units)

        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_expanded)
        )  # (batch, locations, attention_dim)

        attention_weights = tf.nn.softmax(
            self.V(score), axis=1
        )  # (batch, locations, 1)

        context = tf.reduce_sum(
            attention_weights * features, axis=1
        )  # (batch, feature_dim)

        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(vocab_size: int, max_length: int) -> Model:
    """
    Build the attention-based CNN-LSTM image captioning model.

    Architecture:
        Image:   (49, 512) spatial features from VGG block5_pool
        Caption: (max_length,) → Embedding(256) → Dropout(0.3) → LSTM(512)
        Attention: query=LSTM hidden, keys=spatial features → context (512,)
        Merge:   Concatenate(context, LSTM) → Dense(512) → Dropout(0.3) → Dense(vocab)

    Args:
        vocab_size: Vocabulary size (including padding index 0)
        max_length: Maximum caption length in tokens

    Returns:
        Compiled Keras Model
    """
    # ── Image spatial features ──
    image_input = Input(
        shape=(FEATURE_LOCATIONS, FEATURE_DIM), name="image_input"
    )
    
    # FIX: Normalize the raw VGG features! Without this, the massive variance
    # in CNN activations blows up the tanh() in the attention mechanism,
    # causing gradients to vanish and the LSTM to suffer from mode collapse.
    norm_image = tf.keras.layers.BatchNormalization(name="image_norm")(image_input)
    norm_image = tf.keras.layers.Dropout(0.3)(norm_image)

    # ── Caption sequence branch ──
    caption_input = Input(shape=(max_length,), name="caption_input")
    caption_embed = Embedding(
        vocab_size, EMBED_DIM, mask_zero=True, name="caption_embedding"
    )(caption_input)
    caption_drop = Dropout(DROPOUT_RATE, name="caption_dropout")(caption_embed)
    caption_lstm = LSTM(LSTM_UNITS, name="caption_lstm")(caption_drop)

    # ── Attention over spatial features ──
    context = BahdanauAttention(
        ATTENTION_DIM, name="attention"
    )([norm_image, caption_lstm])

    # ── Merge context + LSTM output ──
    merged = Concatenate(name="merge")([context, caption_lstm])
    dense1 = Dense(LSTM_UNITS, activation="relu", name="dense_relu")(merged)
    dense_drop = Dropout(DROPOUT_RATE, name="dense_dropout")(dense1)
    output = Dense(vocab_size, activation="softmax", name="output")(dense_drop)

    # ── Build and compile ──
    model = Model(
        inputs=[image_input, caption_input], outputs=output, name="CaptionIQ"
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
    )

    return model


def print_model_summary(vocab_size: int = 5000, max_length: int = 34):
    """Utility to print the model architecture."""
    model = build_model(vocab_size, max_length)
    model.summary()
    return model


if __name__ == "__main__":
    print_model_summary()
