"""
CaptionIQ — Caption Preprocessing
Downloads Flickr8K dataset, cleans captions, builds vocabulary,
creates train/val/test splits, and saves everything to disk.
"""

import os
import re
import string
import pickle
import kagglehub
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, FLICKR_IMAGES_DIR, FLICKR_TEXT_DIR,
    CAPTIONS_FILE, TOKENIZER_FILE,
    TRAIN_IMAGES_FILE, VAL_IMAGES_FILE, TEST_IMAGES_FILE,
    START_TOKEN, END_TOKEN,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    MIN_WORD_FREQ,
)


def download_flickr8k():
    """
    Download Flickr8K dataset using kagglehub.
    Falls back to manual instructions if kagglehub fails.
    """
    print("=" * 60)
    print("Downloading Flickr8K dataset via kagglehub...")
    print("=" * 60)

    try:
        path = kagglehub.dataset_download("adityajn105/flickr8k")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"\nkagglehub download failed: {e}")
        print("\nPlease download the Flickr8K dataset manually:")
        print("  1. Go to https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("  2. Download and extract to:")
        print(f"     Images → {FLICKR_IMAGES_DIR}")
        print(f"     Captions → {FLICKR_TEXT_DIR}")
        return None


def setup_dataset_dirs(kaggle_path: str = None):
    """
    Ensure dataset directories exist and link downloaded data.
    """
    os.makedirs(FLICKR_IMAGES_DIR, exist_ok=True)
    os.makedirs(FLICKR_TEXT_DIR, exist_ok=True)

    if kaggle_path:
        # kagglehub downloads to a cache dir; copy/link files
        import shutil
        # Find images directory
        for root, dirs, files in os.walk(kaggle_path):
            for f in files:
                src = os.path.join(root, f)
                if f.endswith(".jpg") or f.endswith(".png"):
                    dst = os.path.join(FLICKR_IMAGES_DIR, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                elif f.endswith(".txt") or f.endswith(".csv"):
                    dst = os.path.join(FLICKR_TEXT_DIR, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
        print(f"Images copied to: {FLICKR_IMAGES_DIR}")
        print(f"Text files copied to: {FLICKR_TEXT_DIR}")


def load_raw_captions() -> dict:
    """
    Parse raw captions file into {image_id: [caption1, caption2, ...]}.
    Supports both Flickr8k.token.txt format and captions.txt CSV format.
    """
    captions = {}

    # Try Flickr8k.token.txt format first
    token_file = os.path.join(FLICKR_TEXT_DIR, "Flickr8k.token.txt")
    if os.path.exists(token_file):
        print(f"Loading captions from: {token_file}")
        with open(token_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: image_name#caption_idx\tcaption_text
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                img_caption_id, caption = parts
                image_id = img_caption_id.split("#")[0]
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
        return captions

    # Try captions.txt CSV format (kaggle version)
    captions_csv = os.path.join(FLICKR_TEXT_DIR, "captions.txt")
    if os.path.exists(captions_csv):
        print(f"Loading captions from: {captions_csv}")
        with open(captions_csv, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: image,caption
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                image_id, caption = parts
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
        return captions

    raise FileNotFoundError(
        f"No captions file found. Expected one of:\n"
        f"  {token_file}\n"
        f"  {captions_csv}\n"
        f"Please download the Flickr8K dataset first."
    )


def clean_caption(caption: str) -> str:
    """
    Clean a single caption:
      - Lowercase
      - Remove punctuation / digits / special chars
      - Remove single-character words (except 'a')
      - Strip extra whitespace
    """
    caption = caption.lower()
    # Remove digits
    caption = re.sub(r"\d+", "", caption)
    # Remove punctuation
    caption = caption.translate(str.maketrans("", "", string.punctuation))
    # Remove single characters except 'a'
    caption = " ".join(w for w in caption.split() if len(w) > 1 or w == "a")
    # Strip extra whitespace
    caption = caption.strip()
    return caption


def clean_all_captions(captions: dict) -> dict:
    """
    Clean all captions and add start/end tokens.
    """
    cleaned = {}
    for image_id, caption_list in tqdm(captions.items(), desc="Cleaning captions"):
        cleaned[image_id] = []
        for cap in caption_list:
            cap = clean_caption(cap)
            cap = f"{START_TOKEN} {cap} {END_TOKEN}"
            cleaned[image_id].append(cap)
    return cleaned


def save_captions(captions: dict, filepath: str):
    """
    Save cleaned captions to file (image_id<tab>caption per line).
    """
    with open(filepath, "w") as f:
        for image_id, caption_list in captions.items():
            for cap in caption_list:
                f.write(f"{image_id}\t{cap}\n")
    print(f"Saved cleaned captions to: {filepath}")


def create_splits(captions: dict):
    """
    Split image IDs into train / val / test sets.
    Uses Flickr8K official split files if available, otherwise random split.
    """
    all_images = sorted(captions.keys())

    # Try official split files
    train_file = os.path.join(FLICKR_TEXT_DIR, "Flickr_8k.trainImages.txt")
    val_file = os.path.join(FLICKR_TEXT_DIR, "Flickr_8k.devImages.txt")
    test_file = os.path.join(FLICKR_TEXT_DIR, "Flickr_8k.testImages.txt")

    if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print("Using official Flickr8K split files...")
        train = _load_split_file(train_file, all_images)
        val = _load_split_file(val_file, all_images)
        test = _load_split_file(test_file, all_images)
    else:
        print("Official splits not found — creating random splits...")
        np.random.seed(42)
        np.random.shuffle(all_images)
        train = all_images[:TRAIN_SIZE]
        val = all_images[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
        test = all_images[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    # Save split files
    _save_split(train, TRAIN_IMAGES_FILE, "Train")
    _save_split(val, VAL_IMAGES_FILE, "Val")
    _save_split(test, TEST_IMAGES_FILE, "Test")

    return train, val, test


def _load_split_file(filepath: str, valid_images: list) -> list:
    """Load image IDs from an official split file."""
    valid_set = set(valid_images)
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip() in valid_set]


def _save_split(image_ids: list, filepath: str, name: str):
    """Save a list of image IDs to file."""
    with open(filepath, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")
    print(f"  {name}: {len(image_ids)} images → {filepath}")


def build_tokenizer(captions: dict, train_images: list) -> Tokenizer:
    """
    Fit a Keras Tokenizer on training captions only.
    Filters vocabulary to keep only words with frequency >= MIN_WORD_FREQ.
    Save to disk as pickle.
    """
    # Collect all training captions
    train_captions = []
    train_set = set(train_images)
    for img_id in train_set:
        if img_id in captions:
            train_captions.extend(captions[img_id])

    # Fit tokenizer (word_index is ordered by frequency)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_captions)

    total_words = len(tokenizer.word_index)

    # Filter: keep only words with frequency >= MIN_WORD_FREQ
    freq_words = sum(
        1 for c in tokenizer.word_counts.values()
        if c >= MIN_WORD_FREQ
    )

    print(f"  Total unique words: {total_words}")
    print(f"  Words with freq >= {MIN_WORD_FREQ}: {freq_words}")
    print(f"  Filtered out: {total_words - freq_words} rare words")

    # num_words keeps the top (num_words - 1) words by frequency
    tokenizer.num_words = freq_words + 1  # +1 for padding index 0
    vocab_size = tokenizer.num_words
    print(f"Vocabulary size: {vocab_size}")

    # Compute max caption length
    max_length = max(len(cap.split()) for cap in train_captions)
    print(f"Max caption length: {max_length}")

    # Save tokenizer
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")

    return tokenizer


def main():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  CaptionIQ — Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Download dataset
    kaggle_path = download_flickr8k()
    setup_dataset_dirs(kaggle_path)

    # Step 2: Load and clean captions
    raw_captions = load_raw_captions()
    print(f"Loaded captions for {len(raw_captions)} images")

    cleaned_captions = clean_all_captions(raw_captions)
    save_captions(cleaned_captions, CAPTIONS_FILE)

    # Step 3: Create train/val/test splits
    train, val, test = create_splits(cleaned_captions)

    # Step 4: Build tokenizer on training data
    tokenizer = build_tokenizer(cleaned_captions, train)

    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)
    print(f"  Cleaned captions:  {CAPTIONS_FILE}")
    print(f"  Tokenizer:         {TOKENIZER_FILE}")
    print(f"  Train split:       {len(train)} images")
    print(f"  Val split:         {len(val)} images")
    print(f"  Test split:        {len(test)} images")


if __name__ == "__main__":
    main()
