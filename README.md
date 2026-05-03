---
title: CaptionIQ
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.42.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# CaptionIQ — Automatic Image Captioning

CaptionIQ is an image captioning system trained on the Flickr8K dataset. It builds two VGG-based CNN-LSTM models (VGG16 and VGG19) with Bahdanau attention and supplements them with a fine-tuned BLIP transformer for comparison.

---

## Features

- Dual CNN backbones — VGG16 and VGG19 for spatial feature extraction (49 x 512 maps from 7x7 spatial grid)
- Bahdanau attention LSTM — attends to specific image regions while generating each word
- BLIP integration — Salesforce BLIP transformer for high-quality baseline captions
- Beam search decoding — 15 candidates with confidence scores
- Attention heatmap — gradient saliency overlay showing which regions the model focuses on
- Word cloud — live word distribution across beam search candidates
- Model comparison — VGG16 vs VGG19 vs BLIP side-by-side
- Session history — track all generated captions, export as JSON or CSV
- BLEU evaluation — per-image BLEU-1 through BLEU-4 scoring

---

## Architecture

```
Image → VGG16 / VGG19 block5_pool → (49 x 512) spatial feature map
                                              |
                                  Bahdanau Attention
                                              |
Caption tokens → Embedding(256) → LSTM(256) → Dense → Softmax(vocab)
```

The ensemble averages token-level probability distributions from VGG16 and VGG19 before applying beam search. A knowledge distillation step using BLIP soft targets improves ensemble BLEU-1 by approximately 13%.

---

## Dataset and Training Details

| Parameter              | Value                  |
|------------------------|------------------------|
| Dataset                | Flickr8K               |
| Total images           | 8,000                  |
| Captions per image     | 5                      |
| Total caption pairs    | 40,000                 |
| Train / Test split     | 90% / 10% (7,200 / 800)|
| Vocabulary size        | 8,768 unique words     |
| Max caption length     | 34 tokens              |
| CNN feature size       | 4,096 dimensions       |
| Spatial feature map    | 7 x 7 = 49 regions     |
| LSTM hidden units      | 256                    |
| Beam search width      | 15 candidates          |
| Training epochs        | 50 (early stopping)    |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess dataset

```bash
python src/preprocess.py
```

Cleans captions, builds vocabulary, and creates train/val/test splits.

### 3. Extract features

```bash
python src/extract_features.py --backbone both
```

Extracts spatial feature maps from VGG16 and VGG19 and saves them as `.pkl` files.

### 4. Train models

```bash
python src/train.py --backbone both --epochs 50
```

Trains both VGG16 and VGG19 captioning models. Saves checkpoints and loss plots to `outputs/`.

### 5. Evaluate

```bash
python src/evaluate.py --backbone both
```

Computes corpus BLEU-1 through BLEU-4 on the held-out test set.

### 6. Generate captions

```bash
python src/inference.py --image path/to/image.jpg --backbone vgg16
```

### 7. Launch web app

```bash
streamlit run app.py
```

---

## Project Structure

```
├── src/
│   ├── config.py            # Paths and hyperparameters
│   ├── preprocess.py        # Caption cleaning and tokenization
│   ├── extract_features.py  # VGG feature extraction
│   ├── model.py             # CNN-LSTM with Bahdanau attention
│   ├── model_improved.py    # Improved architecture variant
│   ├── train.py             # Training loop with data generator
│   ├── train_blip.py        # BLIP fine-tuning script
│   ├── inference.py         # Greedy and beam search decoding
│   ├── evaluate.py          # Corpus BLEU evaluation
│   └── utils.py             # Shared utilities
├── outputs/
│   ├── bleu_results.json    # Evaluation results
│   └── loss_vgg*.png        # Training loss curves
├── app.py                   # Streamlit web application
├── requirements.txt
└── README.md
```

Not tracked by git (too large to commit):
- `data/Flickr8k_Dataset/` — 8,000 images (~1 GB)
- `data/*.pkl` — pre-computed VGG feature files (~800 MB each)
- `models/*.h5` — trained Keras model weights (~100 MB each)

---

## Results

Evaluated on the Flickr8K test split (800 images, 5 reference captions each) using corpus BLEU.

| Metric | VGG16 | VGG19 | BLIP   |
|--------|-------|-------|--------|
| BLEU-1 | 0.521 | 0.538 | 0.821  |
| BLEU-4 | 0.098 | —     | 0.312  |

VGG19 outperforms VGG16 on BLEU-1. BLIP substantially outperforms both CNN-LSTM models, which is expected given the scale difference. Knowledge distillation from BLIP soft targets improved the VGG ensemble BLEU-1 by approximately 13%.

---

## Tech Stack

- Deep learning: TensorFlow / Keras
- Feature extraction: VGG16, VGG19 (ImageNet pre-trained weights)
- Transformer: Salesforce BLIP (`Salesforce/blip-image-captioning-base`)
- Text processing: NLTK, Keras Tokenizer
- Evaluation: NLTK corpus BLEU
- Web app: Streamlit
- Dataset: Flickr8K (8,000 images, 5 captions per image)

---

## License

This project was developed for educational purposes as part of a graduate-level computer vision course.
