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
# 🧠 CaptionIQ — AI Image Captioning

> Generate natural language captions for images using VGG16/VGG19 + Bahdanau Attention LSTM on the Flickr8K dataset.

---

## ✨ Features

- **Dual CNN Backbones** — VGG16 and VGG19 for spatial feature extraction (7×7×512)
- **Bahdanau Attention LSTM** — Attends to specific image regions per word
- **Ensemble Mode (BLIP)** — High-quality captions from Salesforce BLIP model
- **Beam Search** — Top-5 diverse captions with confidence bars
- **🔥 Attention Heatmap** — Interactive word-by-word gradient saliency overlay
- **☁️ Word Cloud** — Live word distribution from beam candidates
- **🔄 Model Comparison** — VGG16 vs VGG19 vs Ensemble side-by-side with 🏆 winner
- **📋 Session History** — Track all generated captions, export as JSON/CSV
- **🎲 Surprise Me** — Random Flickr8K image with one click
- **BLEU Evaluation** — Per-image BLEU-1 through BLEU-4 scoring

---

## 🏗️ Architecture

```
Image → VGG16/19 block5_pool → (49 × 512) spatial map
                                      ↓
                          Bahdanau Attention
                                      ↓
Caption tokens → Embedding(256) → LSTM(512) → Softmax(vocab)
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
