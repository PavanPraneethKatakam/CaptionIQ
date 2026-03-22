"""
CaptionIQ — Shared Utility Functions
Load captions, image lists, features, and tokenizer from disk.
"""

import pickle
from typing import Dict, List, Set


def load_captions(filepath: str) -> Dict[str, List[str]]:
    """
    Load cleaned captions from file.
    
    Expected format per line:
        image_id<tab>caption text
    
    Returns:
        dict mapping image_id → list of caption strings
    """
    captions = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            image_id, caption = parts
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions


def load_image_list(filepath: str) -> Set[str]:
    """
    Load a set of image IDs from a text file (one per line).
    """
    with open(filepath, "r") as f:
        return {line.strip() for line in f if line.strip()}


def load_features(pkl_path: str) -> Dict[str, any]:
    """
    Load pre-extracted image features from a pickle file.
    
    Returns:
        dict mapping image_id → numpy array of shape (4096,)
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_tokenizer(pkl_path: str):
    """
    Load a fitted Keras Tokenizer from a pickle file.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def word_for_id(integer: int, tokenizer) -> str:
    """
    Map an integer index back to a word using the tokenizer.
    Uses a cached reverse index for O(1) lookup.
    Returns None if the index is not found or exceeds num_words.
    """
    if not hasattr(tokenizer, '_reverse_index'):
        tokenizer._reverse_index = {
            idx: word for word, idx in tokenizer.word_index.items()
        }
    # Respect vocab filtering
    if tokenizer.num_words is not None and integer >= tokenizer.num_words:
        return None
    return tokenizer._reverse_index.get(integer, None)


def get_vocab_size(tokenizer) -> int:
    """Get vocabulary size, respecting num_words filter if set."""
    if tokenizer.num_words is not None:
        return tokenizer.num_words
    return len(tokenizer.word_index) + 1
