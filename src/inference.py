"""
CaptionIQ — Caption Inference
Greedy search and beam search decoding for generating captions from images.
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

# ── Keras compatibility patch ──────────────────────────────────────
# Models saved with newer Keras include 'quantization_config' in every
# layer config.  Older Keras versions reject the unknown kwarg.
# The Embedding layer's mask_zero=True also serialises a NotEqual op
# that the legacy h5 loader can't resolve by name.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    IMAGE_SIZE, BEAM_WIDTH,
    VGG16_MODEL_FILE, VGG19_MODEL_FILE,
    TOKENIZER_FILE, START_TOKEN, END_TOKEN,
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
    """
    Remove degenerate repetition patterns from generated token list.
    Keeps semantics while avoiding outputs like:
    "dog and dog and dog and ..."
    """
    if not words:
        return []

    # 1) Remove long runs of the same token.
    collapsed = [words[0]]
    for w in words[1:]:
        if len(collapsed) >= 2 and collapsed[-1] == w and collapsed[-2] == w:
            continue
        collapsed.append(w)

    # 2) Break repeated bigram loops.
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
    """
    Penalize incomplete or low-quality endings so beam search prefers
    syntactically complete captions.
    """
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

    # Penalize excessive duplicate words.
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.6:
        penalty += 0.25

    return penalty


def greedy_search(model, tokenizer, feature: np.ndarray, max_length: int) -> str:
    """
    Generate a caption using greedy search (pick highest-probability word each step).

    Args:
        model: Trained captioning model
        tokenizer: Fitted Keras tokenizer
        feature: Image feature vector of shape (4096,)
        max_length: Maximum caption length

    Returns:
        Generated caption string (without start/end tokens)
    """
    in_text = START_TOKEN
    previous_word = None
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")

        # Predict next word probabilities
        yhat = model.predict([np.expand_dims(feature, 0), sequence], verbose=0)[0]
        ranked = np.argsort(yhat)[::-1]

        word = None
        for idx in ranked:
            candidate = word_for_id(int(idx), tokenizer)
            if candidate in (None, START_TOKEN):
                continue
            # Avoid immediate repeated tokens when alternatives exist.
            if previous_word is not None and candidate == previous_word:
                continue
            word = candidate
            break

        if word is None or word == END_TOKEN:
            break

        previous_word = word
        in_text += " " + word

    # Remove start token
    caption_words = in_text.replace(START_TOKEN, "").strip().split()
    caption = " ".join(_clean_caption_tokens(caption_words))
    return caption


def _beam_search_core(predict_next_probs, tokenizer, max_length: int, beam_width: int = 3) -> list:
    """
    Shared beam-search logic.

    Args:
        predict_next_probs: Callable taking padded sequence and returning
            probability vector over vocab.
        tokenizer: Fitted Keras tokenizer
        max_length: Maximum caption length
        beam_width: Number of beams to maintain
    """
    # Each beam: (token_list, cumulative_log_prob)
    start_seq = tokenizer.texts_to_sequences([START_TOKEN])[0]
    if not start_seq:
        return []

    start_id = start_seq[0]
    beams = [(start_seq, 0.0)]
    completed = []
    candidate_pool = max(beam_width * 5, beam_width)
    alpha = 0.7  # Length penalty strength.
    min_words_before_end = 4

    for _ in range(max_length):
        all_candidates = []

        for seq, score in beams:
            # If this sequence already ended, skip
            if len(seq) > 1:
                last_word = word_for_id(seq[-1], tokenizer)
                if last_word == END_TOKEN:
                    completed.append((seq, score))
                    continue

            # Pad and predict
            padded = pad_sequences([seq], maxlen=max_length, padding="post")
            yhat = predict_next_probs(padded)

            # Expand each beam by more than beam_width to keep diverse options.
            top_indices = np.argsort(yhat)[-candidate_pool:]

            for idx in top_indices:
                idx = int(idx)
                word = word_for_id(idx, tokenizer)
                if word is None:
                    continue

                # Block invalid or degenerate transitions.
                if idx == 0 or idx == start_id:
                    continue
                if len(seq) > 1 and idx == seq[-1]:
                    continue
                if word == END_TOKEN and (len(seq) - 1) < min_words_before_end:
                    continue

                # Penalize overusing the same word in one caption.
                repeat_count = sum(1 for token_id in seq if token_id == idx)
                repeat_penalty = 0.15 * repeat_count

                candidate_seq = seq + [idx]
                candidate_score = score + np.log(float(yhat[idx]) + 1e-10) - repeat_penalty
                all_candidates.append((candidate_seq, candidate_score))

        if not all_candidates:
            break

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    # Add remaining beams to completed
    completed.extend(beams)

    # Convert sequences to text
    results = []
    for seq, score in completed:
        words = []
        for idx in seq:
            word = word_for_id(idx, tokenizer)
            if word and word != START_TOKEN and word != END_TOKEN:
                words.append(word)
        caption = " ".join(words)
        if caption:
            cleaned_words = _clean_caption_tokens(words)
            if not cleaned_words:
                continue

            # Length penalty similar to GNMT decoding.
            lp = ((5 + len(cleaned_words)) ** alpha) / ((5 + 1) ** alpha)
            norm_score = score / lp
            norm_score -= _ending_quality_penalty(cleaned_words)
            results.append((" ".join(cleaned_words), norm_score))

    # Sort by score (best first) and deduplicate
    results.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_results = []
    for caption, score in results:
        if caption not in seen:
            seen.add(caption)
            unique_results.append((caption, score))
    unique_results = unique_results[:beam_width]
    confidences = _softmax([score for _, score in unique_results])
    return [
        (caption, confidence)
        for (caption, _), confidence in zip(unique_results, confidences)
    ]


def beam_search(model, tokenizer, feature: np.ndarray,
                max_length: int, beam_width: int = 3) -> list:
    """
    Beam search for a single model.
    """
    def _predict_single(padded_seq):
        return model.predict([np.expand_dims(feature, 0), padded_seq], verbose=0)[0]

    return _beam_search_core(_predict_single, tokenizer, max_length, beam_width)


def beam_search_with_attention(
    model, tokenizer, feature: np.ndarray,
    max_length: int, beam_width: int = 3, target_caption: str = None
) -> tuple:
    """
    Beam search that also records per-step attention weights.
    If target_caption is provided, it extracts attention maps for that exact caption.
    """
    candidates = beam_search(model, tokenizer, feature, max_length, beam_width)
    if not candidates and not target_caption:
        return candidates, []

    # If target_caption is provided, use it. Otherwise use the top candidate.
    best_caption = target_caption if target_caption else candidates[0][0]

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad

        attention_maps = []
        feat_expanded = np.expand_dims(feature, 0)
        
        # We know the exact words we want to trace
        words = best_caption.split()
        in_text = START_TOKEN

        for word in words:
            seq = tokenizer.texts_to_sequences([in_text])[0]
            padded = _pad([seq], maxlen=max_length, padding="post")
            
            # Predict
            feat_tensor = tf.constant(feat_expanded, dtype=tf.float32)
            pad_tensor = tf.constant(padded, dtype=tf.float32)
            
            # Get the word_idx of the actual word in our target caption
            word_seq = tokenizer.texts_to_sequences([word])
            if not word_seq or not word_seq[0]:
                word_idx = 1 # fallback to something safe
            else:
                word_idx = word_seq[0][0]

            with tf.GradientTape() as tape:
                tape.watch(feat_tensor)
                output = model([feat_tensor, pad_tensor], training=False)
                score = output[0, word_idx]
                
            grads = tape.gradient(score, feat_tensor)
            if grads is not None:
                attn_weights = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()
                attn_min, attn_max = attn_weights.min(), attn_weights.max()
                if attn_max > attn_min:
                    attn_weights = (attn_weights - attn_min) / (attn_max - attn_min)
                attention_maps.append((word, attn_weights.reshape(7, 7)))
            else:
                attention_maps.append((word, np.ones((7, 7)) / 49.0))

            in_text += " " + word

        # Ensure candidates are returned correctly if we used a target_caption
        if target_caption:
             # Find the confidence if it was in candidates, otherwise fake it
             conf = 0.5
             for cap, score in candidates:
                 if cap == target_caption:
                     conf = score
                     break
             if (target_caption, conf) in candidates:
                 candidates.remove((target_caption, conf))
             candidates.insert(0, (target_caption, conf))

        return candidates, attention_maps
    except Exception:
        return candidates, []


def beam_search_ensemble_with_attention(
    models: list, tokenizer, features: list,
    max_length: int, beam_width: int = 5
) -> tuple:
    """
    Ensemble beam search with gradient-based attention extraction.
    Returns (captions, attention_maps).
    """
    candidates = beam_search_ensemble(models, tokenizer, features, max_length, beam_width)
    if not candidates or not models:
        return candidates, []
    # Use the first model + first feature for attention visualization
    _, attention_maps = beam_search_with_attention(
        models[0], tokenizer, features[0], max_length, beam_width=1
    )
    return candidates, attention_maps


def beam_search_ensemble(models: list, tokenizer, features: list,
                         max_length: int, beam_width: int = 5) -> list:
    """
    Beam search with model ensembling by averaging next-token probabilities.

    Args:
        models: List of trained captioning models.
        tokenizer: Fitted tokenizer.
        features: List of feature tensors aligned with models.
        max_length: Maximum caption length.
        beam_width: Beam size.
    """
    if not models or len(models) != len(features):
        return []

    def _predict_ensemble(padded_seq):
        probs = []
        for model, feature in zip(models, features):
            p = model.predict([np.expand_dims(feature, 0), padded_seq], verbose=0)[0]
            probs.append(p)
        return np.mean(np.stack(probs, axis=0), axis=0)

    return _beam_search_core(_predict_ensemble, tokenizer, max_length, beam_width)


def extract_single_image_feature(image_path: str, backbone: str = "vgg16") -> np.ndarray:
    """
    Extract feature vector from a single image using VGG.

    Args:
        image_path: Path to the image file
        backbone: "vgg16" or "vgg19"

    Returns:
        Feature vector of shape (4096,)
    """
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


def generate_caption(image_path: str, backbone: str = "vgg16",
                     use_beam: bool = True) -> list:
    """
    Full pipeline: image → feature extraction → caption generation.

    Args:
        image_path: Path to input image
        backbone: "vgg16", "vgg19", or "ensemble"
        use_beam: If True, use beam search; otherwise greedy

    Returns:
        List of (caption, score) tuples
    """
    tokenizer = load_tokenizer(TOKENIZER_FILE)

    if backbone == "ensemble":
        model16 = load_model(VGG16_MODEL_FILE, custom_objects=_CUSTOM_OBJECTS)
        model19 = load_model(VGG19_MODEL_FILE, custom_objects=_CUSTOM_OBJECTS)
        max_length = min(model16.input_shape[1][1], model19.input_shape[1][1])
        feature16 = extract_single_image_feature(image_path, "vgg16")
        feature19 = extract_single_image_feature(image_path, "vgg19")
        if use_beam:
            return beam_search_ensemble(
                [model16, model19], tokenizer, [feature16, feature19], max_length, BEAM_WIDTH
            )
        caption = greedy_search(model19, tokenizer, feature19, max_length)
        return [(caption, 1.0 if caption else 0.0)]

    model_file = VGG16_MODEL_FILE if backbone == "vgg16" else VGG19_MODEL_FILE
    model = load_model(model_file, custom_objects=_CUSTOM_OBJECTS)
    max_length = model.input_shape[1][1]
    feature = extract_single_image_feature(image_path, backbone)
    if use_beam:
        return beam_search(model, tokenizer, feature, max_length, BEAM_WIDTH)
    caption = greedy_search(model, tokenizer, feature, max_length)
    return [(caption, 1.0 if caption else 0.0)]


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--backbone", type=str, default="vgg16",
        choices=["vgg16", "vgg19", "ensemble"],
        help="CNN backbone (default: vgg16)"
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy search instead of beam search"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    print(f"\nGenerating captions for: {args.image}")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"Method: {'Greedy' if args.greedy else f'Beam (width={BEAM_WIDTH})'}")
    print("-" * 40)

    results = generate_caption(args.image, args.backbone, not args.greedy)

    for i, (caption, score) in enumerate(results, 1):
        print(f"  #{i}: {caption}")
        if not args.greedy:
            print(f"       (score: {score:.4f})")

    print()


if __name__ == "__main__":
    main()
