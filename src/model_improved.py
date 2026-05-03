"""
CaptionIQ — Improved Attention-Based Model
Implements step-by-step attention during decoding for better performance.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, Concatenate, Add, LayerNormalization, Layer
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

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model_improved(vocab_size: int, max_length: int) -> Model:
    """
    Improved model with:
    1. LayerNormalization instead of BatchNormalization
    2. Multi-layer LSTM with residual connections
    3. Better regularization
    4. Gradient clipping in optimizer

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

    # Use LayerNormalization instead of BatchNormalization
    # LayerNorm is more stable for pre-extracted features
    norm_image = LayerNormalization(epsilon=1e-6, name="image_norm")(image_input)
    norm_image = Dropout(0.2, name="image_dropout")(norm_image)

    # ── Caption sequence branch with multi-layer LSTM ──
    caption_input = Input(shape=(max_length,), name="caption_input")
    caption_embed = Embedding(
        vocab_size, EMBED_DIM, mask_zero=True, name="caption_embedding"
    )(caption_input)
    caption_drop = Dropout(DROPOUT_RATE, name="caption_dropout")(caption_embed)

    # Layer 1: First LSTM layer
    lstm1 = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_layer1")(caption_drop)
    lstm1 = Dropout(DROPOUT_RATE, name="lstm1_dropout")(lstm1)

    # Layer 2: Second LSTM layer with residual connection
    lstm2 = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_layer2")(lstm1)
    lstm2 = Dropout(DROPOUT_RATE, name="lstm2_dropout")(lstm2)
    lstm2_residual = Add(name="lstm2_residual")([lstm1, lstm2])

    # Layer 3: Final LSTM layer
    caption_lstm = LSTM(LSTM_UNITS, name="caption_lstm")(lstm2_residual)

    # ── Attention over spatial features ──
    context, _ = BahdanauAttention(
        ATTENTION_DIM, name="attention"
    )([norm_image, caption_lstm])

    # ── Merge context + LSTM output ──
    merged = Concatenate(name="merge")([context, caption_lstm])

    # Deeper output layers
    dense1 = Dense(LSTM_UNITS, activation="relu", name="dense_relu1")(merged)
    dense1 = Dropout(DROPOUT_RATE, name="dense_dropout1")(dense1)

    dense2 = Dense(LSTM_UNITS // 2, activation="relu", name="dense_relu2")(dense1)
    dense2 = Dropout(DROPOUT_RATE, name="dense_dropout2")(dense2)

    output = Dense(vocab_size, activation="softmax", name="output")(dense2)

    # ── Build and compile with gradient clipping ──
    model = Model(
        inputs=[image_input, caption_input], outputs=output, name="CaptionIQ_Improved"
    )

    # Add gradient clipping and label smoothing
    from tensorflow.keras.losses import CategoricalCrossentropy

    model.compile(
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=5.0),
    )

    return model


def build_decoder_step_model(vocab_size: int, max_length: int) -> Model:
    """
    Build a single-step decoder for step-by-step inference with attention.
    This model takes the current hidden state and generates the next word.

    This is used during beam search to compute attention at each decoding step,
    not just once at the end.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum caption length (not used but kept for consistency)

    Returns:
        Model that takes (features, hidden, cell, word) and returns (probs, new_hidden, new_cell, attention)
    """
    # Inputs
    image_input = Input(shape=(FEATURE_LOCATIONS, FEATURE_DIM), name="image_input")
    hidden_input = Input(shape=(LSTM_UNITS,), name="hidden_state")
    cell_input = Input(shape=(LSTM_UNITS,), name="cell_state")
    word_input = Input(shape=(1,), dtype=tf.int32, name="word_input")

    # Normalize features
    norm_image = LayerNormalization(epsilon=1e-6)(image_input)

    # Embed current word
    word_embed = Embedding(vocab_size, EMBED_DIM)(word_input)
    word_embed = tf.squeeze(word_embed, axis=1)
    word_embed = Dropout(DROPOUT_RATE)(word_embed)

    # Compute attention using current hidden state
    context, attention_weights = BahdanauAttention(ATTENTION_DIM)([norm_image, hidden_input])

    # Concatenate context + word embedding
    lstm_input = Concatenate()([context, word_embed])
    lstm_input = tf.expand_dims(lstm_input, axis=1)

    # Single LSTM step
    lstm_out, new_hidden, new_cell = LSTM(
        LSTM_UNITS, return_state=True, return_sequences=True
    )(lstm_input, initial_state=[hidden_input, cell_input])

    lstm_out = tf.squeeze(lstm_out, axis=1)
    lstm_out = Dropout(DROPOUT_RATE)(lstm_out)

    # Output layer
    dense = Dense(LSTM_UNITS, activation="relu")(lstm_out)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(vocab_size, activation="softmax")(dense)

    return Model(
        inputs=[image_input, hidden_input, cell_input, word_input],
        outputs=[output, new_hidden, new_cell, attention_weights],
        name="DecoderStep"
    )


def print_model_summary(vocab_size: int = 5000, max_length: int = 34):
    """Utility to print the improved model architecture."""
    print("=" * 60)
    print("IMPROVED MODEL ARCHITECTURE")
    print("=" * 60)
    model = build_model_improved(vocab_size, max_length)
    model.summary()

    print("\n" + "=" * 60)
    print("DECODER STEP MODEL (for step-by-step inference)")
    print("=" * 60)
    decoder = build_decoder_step_model(vocab_size, max_length)
    decoder.summary()

    return model, decoder


if __name__ == "__main__":
    print_model_summary()
