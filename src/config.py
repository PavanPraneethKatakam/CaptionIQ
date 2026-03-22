"""
CaptionIQ — Central Configuration
All hyperparameters, paths, and constants in one place.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Raw dataset paths (Flickr8K)
FLICKR_IMAGES_DIR = os.path.join(DATA_DIR, "Flickr8k_Dataset")
FLICKR_TEXT_DIR = os.path.join(DATA_DIR, "Flickr8k_text")

# Processed data paths
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_clean.txt")
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.pkl")
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, "train_images.txt")
VAL_IMAGES_FILE = os.path.join(DATA_DIR, "val_images.txt")
TEST_IMAGES_FILE = os.path.join(DATA_DIR, "test_images.txt")

# Feature files
VGG16_FEATURES_FILE = os.path.join(DATA_DIR, "vgg16_features.pkl")
VGG19_FEATURES_FILE = os.path.join(DATA_DIR, "vgg19_features.pkl")

# Model checkpoint paths
VGG16_MODEL_FILE = os.path.join(MODELS_DIR, "model_vgg16.h5")
VGG19_MODEL_FILE = os.path.join(MODELS_DIR, "model_vgg19.h5")

# Output paths
VGG16_LOSS_PLOT = os.path.join(OUTPUTS_DIR, "loss_vgg16.png")
VGG19_LOSS_PLOT = os.path.join(OUTPUTS_DIR, "loss_vgg19.png")
BLEU_RESULTS_FILE = os.path.join(OUTPUTS_DIR, "bleu_results.json")

# ──────────────────────────────────────────────
# Image parameters
# ──────────────────────────────────────────────
IMAGE_SIZE = 224  # VGG input requirement

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────
EMBED_DIM = 256           # Word embedding dimension
LSTM_UNITS = 512          # LSTM hidden units
DROPOUT_RATE = 0.3        # Dropout rate (reduced from 0.5)
FEATURE_DIM = 512         # VGG spatial feature channels
FEATURE_LOCATIONS = 49    # 7x7 spatial positions from block5_pool
ATTENTION_DIM = 256       # Bahdanau attention hidden dim

# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
MIN_WORD_FREQ = 5  # Drop words appearing fewer than this many times

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
EPOCHS = 30              # 30 is enough; early stopping handles the rest
BATCH_SIZE = 64          # 2x bigger batch = 2x fewer steps = 2x faster
LEARNING_RATE = 0.001

# Callbacks
EARLY_STOP_PATIENCE = 6   # EarlyStopping patience (epochs)
LR_PATIENCE = 3           # ReduceLROnPlateau patience (epochs)
LR_FACTOR = 0.5           # LR reduction factor

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
BEAM_WIDTH = 5
MAX_LENGTH = 34       # Maximum caption length in tokens

# ──────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────
START_TOKEN = "startseq"
END_TOKEN = "endseq"

# ──────────────────────────────────────────────
# Dataset splits
# ──────────────────────────────────────────────
TRAIN_SIZE = 6000
VAL_SIZE = 1000
TEST_SIZE = 1000

# ──────────────────────────────────────────────
# Create directories if they don't exist
# ──────────────────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)
