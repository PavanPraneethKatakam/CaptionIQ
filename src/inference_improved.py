"""
CaptionIQ — Improved Inference with Step-by-Step Attention
Implements proper attention at each decoding step for better caption quality.
"""

import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.models import Model as KerasModel

# Keras compatibility patch
import keras.src.ops.operation as _keras_op
from keras.src.ops.numpy import NotEqual as _NotEqual

_orig_from_config = _keras_op.Operation.from_config.__func__

@classmethod
def _patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_from_config(cls, config)

_keras_op.Operation.from_config = _patched_from_config

from src.model_improved import BahdanauAttention
_CUSTOM_OBJECTS = {"NotEqual": _NotEqual, "BahdanauAttention": BahdanauAttention}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    IMAGE_SIZE, BEAM_WIDTH,
    VGG16_MODEL_FILE, VGG19_MODEL_FILE,
    TOKENIZER_FILE, START_TOKEN, END_TOKEN, MODELS_DIR,
)
from src.utils import load_tokenizer, word_for_id


def _softmax(values):
    """Stable softmax for confidence normalization."""
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    denom = np.sum(exp_arr)
    if denom <= 0:
        return [0.0] * len(values)
    return (exp_arr / denom).tolist()


def _clean_caption_tokens(words):
    """Remove degenerate repetition patterns."""
    if not words:
        return []

    # Remove long runs of the same token
    collapsed = [words[0]]
    for w in words[1:]:
        if len(collapsed) >= 2 and collapsed[-1] == w and collapsed[-2] == w:
            continue
        collapsed.append(w)

    # Break repeated bigram loops
    cleaned = []
    for w in collapsed:
        cleaned.append(w)
        if len(cleaned) >= 6:
            b1 = tuple(cleaned[-2:])
            b2 = tuple(cleaned[-4:-2])
            b3 = tuple(cleaned[-6:-4])
            if b1 == b2 == b3:
                cleaned = cleaned[:-2]
                break

    return cleaned


def _ending_quality_penalty(words):
    """Penalize incomplete or low-quality endings."""
    if not words:
        return 2.0

    penalty = 0.0
    weak_endings = {
        "a", "an", "the", "in", "on", "at", "of", "to", "for",
        "with", "by", "from", "and", "or", "but", "as",
    }

    if words[-1] in weak_endings:
        penalty += 0.9
    if len(words) < 5:
        penalty += 0.35

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.6:
        penalty += 0.25

    return penalty


def beam_search_improved(model, tokenizer, feature: np.ndarray,
                        max_length: int, beam_width: int = 5) -> list:
    """
    Improved beam search with better diversity and quality control.

    Key improvements over baseline:
    1. Larger candidate pool for diversity
    2. Better repetition penalties
    3. Length normalization (GNMT-style)
    4. Quality-based ending penalties
    """
    start_seq = tokenizer.texts_to_sequences([START_TOKEN])[0]
    if not start_seq:
        return []

    start_id = start_seq[0]
    beams = [(start_seq, 0.0)]
    completed = []

    # Larger candidate pool for more diversity
    candidate_pool = max(beam_width * 8, 20)
    alpha = 0.7  # Length penalty strength
    min_words_before_end = 5  # Require at least 5 words

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            # Check if sequence already ended
            if len(seq) > 1:
                last_word = word_for_id(seq[-1], tokenizer)
                if last_word == END_TOKEN:
                    completed.append((seq, score))
                    continue

            # Pad and predict
            padded = pad_sequences([seq], maxlen=max_length, padding="post")
            yhat = model.predict([np.expand_dims(feature, 0), padded], verbose=0)[0]

            # Get top candidates
            top_indices = np.argsort(yhat)[-candidate_pool:]

            for idx in top_indices:
                idx = int(idx)
                word = word_for_id(idx, tokenizer)
                if word is None:
                    continue

                # Block invalid transitions
                if idx == 0 or idx == start_id:
                    continue

                # Prevent immediate repetition
                if len(seq) > 1 and idx == seq[-1]:
                    continue

                # Don't end too early
                if word == END_TOKEN and (len(seq) - 1) < min_words_before_end:
                    continue

                # Stronger repetition penalty
                repeat_count = sum(1 for token_id in seq if token_id == idx)
                repeat_penalty = 0.2 * repeat_count  # Increased from 0.15

                # Diversity bonus for rare words (encourages descriptive captions)
                prob = float(yhat[idx])
                diversity_bonus = 0.0
                if prob < 0.1 and prob > 0.01:  # Rare but not too rare
                    diversity_bonus = 0.05

                candidate_seq = seq + [idx]
                candidate_score = (
                    score +
                    np.log(prob + 1e-10) -
                    repeat_penalty +
                    diversity_bonus
                )
                all_candidates.append((candidate_seq, candidate_score))

        if not all_candidates:
            break

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    # Add remaining beams to completed
    completed.extend(beams)

    # Convert sequences to text with quality scoring
    results = []
    for seq, score in completed:
        words = []
        for idx in seq:
            word = word_for_id(idx, tokenizer)
            if word and word != START_TOKEN and word != END_TOKEN:
                words.append(word)

        if not words:
            continue

        cleaned_words = _clean_caption_tokens(words)
        if not cleaned_words:
            continue

        # GNMT-style length penalty
        lp = ((5 + len(cleaned_words)) ** alpha) / ((5 + 1) ** alpha)
        norm_score = score / lp

        # Apply ending quality penalty
        norm_score -= _ending_quality_penalty(cleaned_words)

        results.append((" ".join(cleaned_words), norm_score))

    # Sort and deduplicate
    results.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_results = []
    for caption, score in results:
        if caption not in seen:
            seen.add(caption)
            unique_results.append((caption, score))

    unique_results = unique_results[:beam_width]

    # Normalize confidences to [0, 1]
    confidences = _softmax([score for _, score in unique_results])
    return [
        (caption, confidence)
        for (caption, _), confidence in zip(unique_results, confidences)
    ]


def extract_single_image_feature(image_path: str, backbone: str = "vgg16") -> np.ndarray:
    """Extract feature vector from a single image using VGG."""
    if backbone == "vgg16":
        base_model = VGG16(weights="imagenet")
        preprocess_fn = vgg16_preprocess
    else:
        base_model = VGG19(weights="imagenet")
        preprocess_fn = vgg19_preprocess

    model = KerasModel(
        inputs=base_model.input,
        outputs=base_model.get_layer("block5_pool").output
    )

    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_fn(image)

    feature = model.predict(image, verbose=0)[0]  # (7, 7, 512)
    h, w, c = feature.shape
    return feature.reshape(h * w, c)  # (49, 512)


def generate_caption_improved(image_path: str, backbone: str = "vgg16",
                             beam_width: int = None) -> list:
    """
    Generate captions using the improved model.

    Args:
        image_path: Path to input image
        backbone: "vgg16" or "vgg19"
        beam_width: Beam search width (default: 5)

    Returns:
        List of (caption, confidence) tuples
    """
    beam_width = beam_width or BEAM_WIDTH
    tokenizer = load_tokenizer(TOKENIZER_FILE)

    # Load improved model
    if backbone == "vgg16":
        model_file = os.path.join(MODELS_DIR, "model_vgg16_improved.h5")
        fallback_file = VGG16_MODEL_FILE
    else:
        model_file = os.path.join(MODELS_DIR, "model_vgg19_improved.h5")
        fallback_file = VGG19_MODEL_FILE

    # Try improved model first, fall back to baseline
    if os.path.exists(model_file):
        print(f"Using improved model: {model_file}")
        model = load_model(model_file, custom_objects=_CUSTOM_OBJECTS)
    elif os.path.exists(fallback_file):
        print(f"Improved model not found, using baseline: {fallback_file}")
        model = load_model(fallback_file, custom_objects=_CUSTOM_OBJECTS)
    else:
        raise FileNotFoundError(f"No model found for {backbone}")

    max_length = model.input_shape[1][1]
    feature = extract_single_image_feature(image_path, backbone)

    return beam_search_improved(model, tokenizer, feature, max_length, beam_width)


def compare_models(image_path: str, backbone: str = "vgg19"):
    """
    Compare baseline vs improved model on the same image.
    """
    print("=" * 70)
    print(f"COMPARING BASELINE vs IMPROVED MODEL ({backbone.upper()})")
    print("=" * 70)
    print(f"Image: {image_path}\n")

    tokenizer = load_tokenizer(TOKENIZER_FILE)
    feature = extract_single_image_feature(image_path, backbone)

    # Load both models
    baseline_file = VGG16_MODEL_FILE if backbone == "vgg16" else VGG19_MODEL_FILE
    improved_file = os.path.join(
        MODELS_DIR,
        f"model_{backbone}_improved.h5"
    )

    if not os.path.exists(improved_file):
        print(f"❌ Improved model not found: {improved_file}")
        print("   Train it first: python src/train_improved.py --backbone", backbone)
        return

    print("Loading models...")
    baseline_model = load_model(baseline_file, custom_objects=_CUSTOM_OBJECTS)
    improved_model = load_model(improved_file, custom_objects=_CUSTOM_OBJECTS)
    max_length = baseline_model.input_shape[1][1]

    # Generate captions with both
    print("\n" + "-" * 70)
    print("BASELINE MODEL")
    print("-" * 70)
    from src.inference import beam_search
    baseline_results = beam_search(baseline_model, tokenizer, feature, max_length, BEAM_WIDTH)
    for i, (caption, conf) in enumerate(baseline_results, 1):
        print(f"  {i}. {caption}")
        print(f"     Confidence: {conf:.3f}\n")

    print("-" * 70)
    print("IMPROVED MODEL")
    print("-" * 70)
    improved_results = beam_search_improved(improved_model, tokenizer, feature, max_length, BEAM_WIDTH)
    for i, (caption, conf) in enumerate(improved_results, 1):
        print(f"  {i}. {caption}")
        print(f"     Confidence: {conf:.3f}\n")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate captions with improved model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--backbone", type=str, default="vgg19",
        choices=["vgg16", "vgg19"],
        help="CNN backbone (default: vgg19)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare baseline vs improved model"
    )
    parser.add_argument(
        "--beam-width", type=int, default=BEAM_WIDTH,
        help=f"Beam search width (default: {BEAM_WIDTH})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    if args.compare:
        compare_models(args.image, args.backbone)
    else:
        print(f"\nGenerating captions for: {args.image}")
        print(f"Backbone: {args.backbone.upper()}")
        print(f"Beam width: {args.beam_width}")
        print("-" * 60)

        results = generate_caption_improved(args.image, args.backbone, args.beam_width)

        for i, (caption, score) in enumerate(results, 1):
            print(f"  #{i}: {caption}")
            print(f"       Confidence: {score:.3f}\n")


if __name__ == "__main__":
    main()
