"""
CaptionIQ — Improved Training Script
Includes better hyperparameters and training strategies.
"""

import os
import sys
import argparse
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    EPOCHS, BATCH_SIZE, MAX_LENGTH, FEATURE_DIM, FEATURE_LOCATIONS,
    VGG16_FEATURES_FILE, VGG19_FEATURES_FILE,
    CAPTIONS_FILE, TOKENIZER_FILE,
    TRAIN_IMAGES_FILE, VAL_IMAGES_FILE,
    MODELS_DIR, OUTPUTS_DIR,
)
from src.utils import load_captions, load_image_list, load_features, load_tokenizer, get_vocab_size
from src.model_improved import build_model_improved, BahdanauAttention


def data_generator(captions: dict, features: dict, tokenizer,
                   max_length: int, vocab_size: int, batch_size: int):
    """
    Generator that yields batches of ((image_features, partial_sequence), next_word).
    """
    image_ids = list(captions.keys())

    while True:
        np.random.shuffle(image_ids)
        img_batch, seq_batch, target_batch = [], [], []

        for img_id in image_ids:
            if img_id not in features:
                continue

            img_feature = features[img_id]

            for caption in captions[img_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length, padding="post")[0]
                    out_word = to_categorical([seq[i]], num_classes=vocab_size)[0]

                    img_batch.append(img_feature)
                    seq_batch.append(in_seq)
                    target_batch.append(out_word)

                    if len(img_batch) >= batch_size:
                        yield (
                            (np.array(img_batch, dtype=np.float32),
                             np.array(seq_batch, dtype=np.int32)),
                            np.array(target_batch, dtype=np.float32)
                        )
                        img_batch, seq_batch, target_batch = [], [], []

        if img_batch:
            yield (
                (np.array(img_batch, dtype=np.float32),
                 np.array(seq_batch, dtype=np.int32)),
                np.array(target_batch, dtype=np.float32)
            )


def make_dataset(captions, features, tokenizer, max_length, vocab_size, batch_size):
    """Wrap the Python generator in a tf.data.Dataset."""
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(captions, features, tokenizer,
                               max_length, vocab_size, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, FEATURE_LOCATIONS, FEATURE_DIM), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32),
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def count_steps(captions: dict, features: dict, tokenizer, batch_size: int) -> int:
    """Count total training steps per epoch."""
    total = 0
    for img_id in captions:
        if img_id not in features:
            continue
        for caption in captions[img_id]:
            seq = tokenizer.texts_to_sequences([caption])[0]
            total += len(seq) - 1
    return max(1, math.ceil(total / batch_size))


def compute_max_length(captions: dict) -> int:
    """Compute maximum caption length from the data."""
    max_len = 0
    for caps in captions.values():
        for cap in caps:
            length = len(cap.split())
            if length > max_len:
                max_len = length
    return max_len


def cosine_decay_with_warmup(epoch, total_epochs=50, warmup_epochs=5,
                              initial_lr=0.0003, min_lr=1e-6):
    """
    Cosine annealing learning rate schedule with warmup.
    Better than step decay for deep learning.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def plot_loss(history: dict, filepath: str, title: str):
    """Plot and save training loss curve."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss", color="#4a90d9", linewidth=2)
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss", color="#e74c3c", linewidth=2)
    plt.title(f"{title} - Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Learning rate plot
    plt.subplot(1, 2, 2)
    if "lr" in history:
        plt.plot(history["lr"], label="Learning Rate", color="#2ecc71", linewidth=2)
        plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.yscale("log")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Loss plot saved to: {filepath}")


def train_improved(backbone: str = "vgg16", epochs: int = None, resume: bool = True):
    """
    Train the improved captioning model.

    Args:
        backbone: "vgg16" or "vgg19"
        epochs: Override default epoch count
        resume: Resume from checkpoint if available
    """
    epochs = epochs or 50  # Train longer with improved model

    # Select files based on backbone
    if backbone == "vgg16":
        features_file = VGG16_FEATURES_FILE
        model_file = os.path.join(MODELS_DIR, "model_vgg16_improved.h5")
        loss_plot = os.path.join(OUTPUTS_DIR, "loss_vgg16_improved.png")
    else:
        features_file = VGG19_FEATURES_FILE
        model_file = os.path.join(MODELS_DIR, "model_vgg19_improved.h5")
        loss_plot = os.path.join(OUTPUTS_DIR, "loss_vgg19_improved.png")

    print("=" * 60)
    print(f"  CaptionIQ IMPROVED — Training with {backbone.upper()}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    all_captions = load_captions(CAPTIONS_FILE)
    tokenizer = load_tokenizer(TOKENIZER_FILE)
    features = load_features(features_file)
    train_images = load_image_list(TRAIN_IMAGES_FILE)
    val_images = load_image_list(VAL_IMAGES_FILE)

    vocab_size = get_vocab_size(tokenizer)
    print(f"  Vocabulary size: {vocab_size}")

    train_captions = {k: v for k, v in all_captions.items() if k in train_images}
    val_captions = {k: v for k, v in all_captions.items() if k in val_images}
    print(f"  Train images: {len(train_captions)}")
    print(f"  Val images:   {len(val_captions)}")

    max_length = compute_max_length(train_captions)
    print(f"  Max caption length: {max_length}")

    # Build or resume model
    if resume and os.path.exists(model_file):
        print(f"\nLoading checkpoint: {model_file}")
        try:
            model = load_model(
                model_file,
                custom_objects={"BahdanauAttention": BahdanauAttention}
            )
            print("  Successfully loaded checkpoint")
        except Exception as exc:
            print(f"  Warning: could not load checkpoint ({exc})")
            print("  Building new model...")
            model = build_model_improved(vocab_size, max_length)
    else:
        print("\nBuilding improved model...")
        model = build_model_improved(vocab_size, max_length)

    model.summary()

    # Create datasets
    train_dataset = make_dataset(
        train_captions, features, tokenizer, max_length, vocab_size, BATCH_SIZE
    )
    val_dataset = make_dataset(
        val_captions, features, tokenizer, max_length, vocab_size, BATCH_SIZE
    )

    train_steps = count_steps(train_captions, features, tokenizer, BATCH_SIZE)
    val_steps = count_steps(val_captions, features, tokenizer, BATCH_SIZE)
    print(f"  Train steps/epoch: {train_steps}")
    print(f"  Val steps/epoch:   {val_steps}")

    # Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            model_file, monitor="val_loss", save_best_only=True, verbose=1
        ),
        LearningRateScheduler(
            lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=epochs),
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
    ]

    # Train
    print(f"\nTraining for {epochs} epochs with improved architecture...")
    print("Improvements:")
    print("  ✓ Multi-layer LSTM with residual connections")
    print("  ✓ LayerNormalization instead of BatchNormalization")
    print("  ✓ Label smoothing (0.1)")
    print("  ✓ Gradient clipping (norm=5.0)")
    print("  ✓ Cosine learning rate schedule with warmup")
    print("  ✓ Deeper output layers")
    print()

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(model_file)
    print(f"\nModel saved to: {model_file}")

    # Plot loss
    plot_loss(
        history.history, loss_plot,
        f"CaptionIQ Improved — {backbone.upper()}"
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train improved CaptionIQ model")
    parser.add_argument(
        "--backbone", type=str, default="vgg19",
        choices=["vgg16", "vgg19", "both"],
        help="Which backbone features to use (default: vgg19)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from checkpoint if available (default: enabled)"
    )
    args = parser.parse_args()

    backbones = ["vgg16", "vgg19"] if args.backbone == "both" else [args.backbone]

    for backbone in backbones:
        train_improved(backbone, args.epochs, args.resume)

    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluate.py --backbone vgg19 --improved")
    print("  2. Compare with baseline: python src/evaluate.py --backbone vgg19")
    print("  3. Test inference: python src/inference.py --image <path> --backbone vgg19 --improved")


if __name__ == "__main__":
    main()
