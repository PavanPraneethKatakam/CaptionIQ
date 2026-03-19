---
title: CaptionIQ
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
python_version: "3.10"
app_file: app.py
pinned: false
---
# 🧠 CaptionIQ — AI Image Captioning

> Generate natural language captions for images using VGG16/VGG19 + LSTM on the Flickr8K dataset.

---

## ✨ Features

- **Dual CNN Backbones** — VGG16 and VGG19 for image feature extraction
- **LSTM Caption Decoder** — Generates fluent, descriptive captions
- **Beam Search** — Top-3 diverse captions with confidence scores
- **BLEU Evaluation** — BLEU-1 through BLEU-4 with VGG16 vs VGG19 comparison
- **Streamlit Web App** — Upload images, switch backbones, download captions
- **Demo Mode** — Try with preloaded Flickr8K sample images

---

## 🏗️ Architecture

```
Image → VGG16/19 (fc2) → 4096-d → Dense(256) ─┐
                                                 ├→ Add → Dense(256) → Softmax(vocab)
Caption → Embedding(256) → LSTM(256) ───────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Dataset

```bash
python src/preprocess.py
```

Downloads Flickr8K, cleans captions, builds vocabulary, creates train/val/test splits.

### 3. Extract Features

```bash
python src/extract_features.py --backbone both
```

Extracts 4096-d features from VGG16 and VGG19 (saved as `.pkl`).

### 4. Train Models

```bash
python src/train.py --backbone both --epochs 20
```

Trains both VGG16 and VGG19 captioning models. Saves checkpoints and loss plots.

### 5. Evaluate

```bash
python src/evaluate.py --backbone both
```

Computes BLEU-1 to BLEU-4 on the test set. Prints VGG16 vs VGG19 comparison table.

### 6. Generate Captions

```bash
python src/inference.py --image path/to/image.jpg --backbone vgg16
```

### 7. Launch Web App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── data/                    # Dataset & preprocessed files
├── models/                  # Trained model checkpoints (.h5)
├── outputs/                 # Loss plots, BLEU results
├── src/
│   ├── config.py            # Paths & hyperparameters
│   ├── preprocess.py        # Caption cleaning & tokenization
│   ├── extract_features.py  # VGG feature extraction
│   ├── model.py             # CNN-LSTM architecture
│   ├── train.py             # Training with data generator
│   ├── inference.py         # Greedy & beam search
│   ├── evaluate.py          # BLEU score evaluation
│   └── utils.py             # Shared utilities
├── app.py                   # Streamlit web app
├── requirements.txt         # Dependencies
└── README.md
```

---

## 📊 Results

| Metric  | VGG16  | VGG19  |
|---------|--------|--------|
| BLEU-1  | —      | —      |
| BLEU-2  | —      | —      |
| BLEU-3  | —      | —      |
| BLEU-4  | —      | —      |

> Results will be populated after training and evaluation.

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Feature Extraction**: VGG16, VGG19 (ImageNet pretrained)
- **Text Processing**: NLTK, Keras Tokenizer
- **Evaluation**: NLTK BLEU
- **Web App**: Streamlit
- **Dataset**: Flickr8K (8,000 images, 5 captions each)

---

## 📄 License

This project is for educational purposes.
