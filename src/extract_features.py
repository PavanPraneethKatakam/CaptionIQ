"""
CaptionIQ — Image Feature Extraction
Extract spatial feature maps from VGG16 and VGG19 (`block5_pool`).
Save features as pickle files for training.
"""

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FLICKR_IMAGES_DIR, IMAGE_SIZE,
    VGG16_FEATURES_FILE, VGG19_FEATURES_FILE,
)


def build_feature_extractor(backbone: str = "vgg16") -> tuple:
    """
    Build a feature extractor from a pre-trained VGG model.
    Outputs `block5_pool` spatial features (7x7x512).

    Args:
        backbone: "vgg16" or "vgg19"

    Returns:
        (model, preprocess_fn) tuple
    """
    if backbone == "vgg16":
        base_model = VGG16(weights="imagenet")
        preprocess_fn = vgg16_preprocess
    elif backbone == "vgg19":
        base_model = VGG19(weights="imagenet")
        preprocess_fn = vgg19_preprocess
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use 'vgg16' or 'vgg19'.")

    # Use block5_pool for spatial features (7x7x512) instead of fc2 (4096)
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("block5_pool").output
    )
    print(f"\n{backbone.upper()} feature extractor loaded")
    print(f"  Output shape: {model.output_shape}  (spatial features)")
    return model, preprocess_fn


def extract_features(model, preprocess_fn, images_dir: str) -> dict:
    """
    Extract features for all images in a directory.

    Returns:
        dict mapping filename → numpy array of shape (49, 512)
    """
    features = {}
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Extracting features for {len(image_files)} images...")
    for fname in tqdm(image_files, desc="Extracting"):
        filepath = os.path.join(images_dir, fname)
        try:
            # Load and preprocess image
            image = load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_fn(image)

            # Extract spatial feature map and reshape to (49, 512)
            feature = model.predict(image, verbose=0)[0]  # (7, 7, 512)
            h, w, c = feature.shape
            features[fname] = feature.reshape(h * w, c)  # (49, 512)
        except Exception as e:
            print(f"  Warning: Failed to process {fname}: {e}")

    print(f"Extracted features for {len(features)} images")
    return features


def save_features(features: dict, filepath: str):
    """Save features dict to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(features, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Features saved to: {filepath} ({size_mb:.1f} MB)")


def main():
    """Extract features using VGG16 and/or VGG19."""
    parser = argparse.ArgumentParser(description="Extract VGG features from images")
    parser.add_argument(
        "--backbone", type=str, default="vgg19",
        choices=["vgg16", "vgg19", "both"],
        help="Which backbone to use for extraction (default: vgg19)"
    )
    args = parser.parse_args()

    if not os.path.exists(FLICKR_IMAGES_DIR):
        print(f"Error: Image directory not found: {FLICKR_IMAGES_DIR}")
        print("Please run preprocess.py first to download the dataset.")
        return

    backbones = ["vgg16", "vgg19"] if args.backbone == "both" else [args.backbone]

    for backbone in backbones:
        print("\n" + "=" * 60)
        print(f"  Extracting {backbone.upper()} features")
        print("=" * 60)

        model, preprocess_fn = build_feature_extractor(backbone)
        features = extract_features(model, preprocess_fn, FLICKR_IMAGES_DIR)

        output_file = VGG16_FEATURES_FILE if backbone == "vgg16" else VGG19_FEATURES_FILE
        save_features(features, output_file)

    print("\n✓ Feature extraction complete!")


if __name__ == "__main__":
    main()
