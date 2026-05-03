"""
CaptionIQ — BLEU Score Evaluation
Generate captions for the test set and compute BLEU-1 through BLEU-4 scores.
Compare VGG16 vs VGG19 performance.
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CAPTIONS_FILE, TOKENIZER_FILE,
    VGG16_FEATURES_FILE, VGG19_FEATURES_FILE,
    VGG16_MODEL_FILE, VGG19_MODEL_FILE,
    TEST_IMAGES_FILE, BLEU_RESULTS_FILE,
    START_TOKEN, END_TOKEN, MAX_LENGTH,
)
from src.utils import load_captions, load_image_list, load_features, load_tokenizer, get_vocab_size
from src.inference import greedy_search, beam_search
from src.config import BEAM_WIDTH
from tensorflow.keras.models import load_model

# ── Keras compatibility patch ──────────────────────────────────────
# Models saved with newer Keras include 'quantization_config' in every
# layer config.  Older Keras versions reject the unknown kwarg.
# The Embedding layer's mask_zero=True also serialises a NotEqual op
# that the legacy h5 loader can't resolve by name.
# Fix: (1) monkey-patch from_config to strip quantization_config,
#      (2) import NotEqual so we can register it as a custom object.
import keras.src.ops.operation as _keras_op
from keras.src.ops.numpy import NotEqual as _NotEqual

_orig_from_config = _keras_op.Operation.from_config.__func__


@classmethod  # type: ignore[misc]
def _patched_from_config(cls, config):  # noqa: N805
    config.pop("quantization_config", None)
    return _orig_from_config(cls, config)


_keras_op.Operation.from_config = _patched_from_config

from src.model import BahdanauAttention
_CUSTOM_OBJECTS = {"NotEqual": _NotEqual, "BahdanauAttention": BahdanauAttention}
# ───────────────────────────────────────────────────────────────────


def get_references(captions: dict, image_ids: set) -> dict:
    """
    Extract reference captions for evaluation.
    Removes start/end tokens and returns as list of word lists.

    Returns:
        dict mapping image_id → list of reference word lists
    """
    references = {}
    for img_id in image_ids:
        if img_id not in captions:
            continue
        refs = []
        for cap in captions[img_id]:
            # Remove start/end tokens and split into words
            words = [
                w for w in cap.split()
                if w not in (START_TOKEN, END_TOKEN)
            ]
            refs.append(words)
        references[img_id] = refs
    return references


def evaluate_model(backbone: str, captions: dict, features: dict,
                   tokenizer, test_images: set) -> dict:
    """
    Evaluate a trained model on the test set.

    Args:
        backbone: "vgg16" or "vgg19"
        captions: All cleaned captions
        features: Pre-extracted image features
        tokenizer: Fitted tokenizer
        test_images: Set of test image IDs

    Returns:
        dict with BLEU-1 through BLEU-4 scores
    """
    model_file = VGG16_MODEL_FILE if backbone == "vgg16" else VGG19_MODEL_FILE

    if not os.path.exists(model_file):
        print(f"  Warning: Model not found: {model_file}")
        return None

    print(f"\nLoading {backbone.upper()} model...")
    model = load_model(model_file, custom_objects=_CUSTOM_OBJECTS)

    # Determine max_length from model
    max_length = model.input_shape[1][1]

    # Get references
    references = get_references(captions, test_images)

    # Generate captions for test images
    actual_refs = []
    hypotheses = []
    skipped = 0

    print(f"Generating captions for {len(test_images)} test images...")
    for img_id in tqdm(sorted(test_images), desc=f"Evaluating {backbone.upper()}"):
        if img_id not in features or img_id not in references:
            skipped += 1
            continue

        feature = features[img_id]

        # Generate caption using beam search for higher quality
        beam_results = beam_search(model, tokenizer, feature, max_length, BEAM_WIDTH)
        caption = beam_results[0][0] if beam_results else ""
        hypothesis = caption.split()

        hypotheses.append(hypothesis)
        actual_refs.append(references[img_id])

    if skipped > 0:
        print(f"  Skipped {skipped} images (missing features or references)")

    print(f"  Evaluated {len(hypotheses)} images")

    # ── Compute BLEU scores ──
    smooth = SmoothingFunction().method1

    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = corpus_bleu(actual_refs, hypotheses, weights=weights,
                            smoothing_function=smooth)
        bleu_scores[f"BLEU-{n}"] = round(score, 4)
        print(f"  BLEU-{n}: {score:.4f}")

    return bleu_scores


def print_comparison_table(results: dict):
    """Print a formatted comparison table of VGG16 vs VGG19."""
    print("\n" + "=" * 55)
    print("  VGG16 vs VGG19 — BLEU Score Comparison")
    print("=" * 55)
    print(f"{'Metric':<10} {'VGG16':>10} {'VGG19':>10} {'Diff':>10}")
    print("-" * 55)

    for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]:
        v16 = results.get("vgg16", {}).get(metric, "N/A")
        v19 = results.get("vgg19", {}).get(metric, "N/A")

        if isinstance(v16, float) and isinstance(v19, float):
            diff = v19 - v16
            sign = "+" if diff >= 0 else ""
            print(f"{metric:<10} {v16:>10.4f} {v19:>10.4f} {sign}{diff:>9.4f}")
        else:
            print(f"{metric:<10} {str(v16):>10} {str(v19):>10} {'—':>10}")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CaptionIQ with BLEU scores")
    parser.add_argument(
        "--backbone", type=str, default="vgg19",
        choices=["vgg16", "vgg19", "both"],
        help="Which model(s) to evaluate (default: vgg19)"
    )
    args = parser.parse_args()

    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("=" * 60)
    print("  CaptionIQ — BLEU Score Evaluation")
    print("=" * 60)

    # Load shared data
    all_captions = load_captions(CAPTIONS_FILE)
    tokenizer = load_tokenizer(TOKENIZER_FILE)
    test_images = load_image_list(TEST_IMAGES_FILE)

    print(f"Test images: {len(test_images)}")

    backbones = ["vgg16", "vgg19"] if args.backbone == "both" else [args.backbone]
    results = {}

    for backbone in backbones:
        features_file = VGG16_FEATURES_FILE if backbone == "vgg16" else VGG19_FEATURES_FILE

        if not os.path.exists(features_file):
            print(f"\n  Warning: Features not found: {features_file}")
            continue

        features = load_features(features_file)
        scores = evaluate_model(backbone, all_captions, features, tokenizer, test_images)

        if scores:
            results[backbone] = scores

    # Save results
    if results:
        with open(BLEU_RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {BLEU_RESULTS_FILE}")

    # Print comparison table if both models evaluated
    if "vgg16" in results and "vgg19" in results:
        print_comparison_table(results)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
