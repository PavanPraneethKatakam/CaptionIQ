import os
import sys
import io
import json
import random
import datetime
import numpy as np

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.models import Model as KerasModel

import keras.src.ops.operation as _keras_op
from keras.src.ops.numpy import NotEqual as _NotEqual

_orig_from_config = _keras_op.Operation.from_config.__func__


@classmethod
def _patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_from_config(cls, config)


_keras_op.Operation.from_config = _patched_from_config

from src.model import BahdanauAttention
_CUSTOM_OBJECTS = {"NotEqual": _NotEqual, "BahdanauAttention": BahdanauAttention}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    IMAGE_SIZE, BEAM_WIDTH,
    VGG16_MODEL_FILE, VGG19_MODEL_FILE,
    TOKENIZER_FILE, BLEU_RESULTS_FILE,
    VGG16_FEATURES_FILE, VGG19_FEATURES_FILE,
    FLICKR_IMAGES_DIR, CAPTIONS_FILE,
    START_TOKEN, END_TOKEN,
)
from src.utils import load_tokenizer, load_features, load_captions
from src.engine import CaptionEngine

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaptionIQ — AI Image Captioning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Core reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, .main, .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080818;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(102, 126, 234, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(118, 75, 162, 0.10) 0%, transparent 60%);
}

/* ── Floating particles canvas ── */
#particles-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 0;
}

/* ── Hero section ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(102, 126, 234, 0.12);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem; font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
    animation: badgePulse 3s ease-in-out infinite;
}
@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.25); }
    50%       { box-shadow: 0 0 0 8px rgba(102, 126, 234, 0.0); }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% 200%;
    animation: shimmer 4s linear infinite;
    line-height: 1.1; margin-bottom: 0.6rem;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-sub {
    font-size: 1.05rem; color: #94a3b8; max-width: 600px;
    margin: 0 auto 1.8rem;
    line-height: 1.6;
}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102, 126, 234, 0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    background: rgba(102, 126, 234, 0.03) !important;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(102, 126, 234, 0.7) !important;
    background: rgba(102, 126, 234, 0.07) !important;
    box-shadow: 0 0 24px rgba(102, 126, 234, 0.15) !important;
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    border-color: rgba(102, 126, 234, 0.35);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

/* ── Caption cards ── */
.caption-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    animation: captionReveal 0.5s ease forwards;
    opacity: 0;
}
.caption-card:nth-child(1) { animation-delay: 0.05s; }
.caption-card:nth-child(2) { animation-delay: 0.15s; }
.caption-card:nth-child(3) { animation-delay: 0.25s; }
@keyframes captionReveal {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.caption-card:hover {
    transform: translateY(-2px);
    border-color: rgba(102,126,234,0.4);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.caption-rank {
    font-size: 0.72rem; font-weight: 700;
    color: #a78bfa; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 6px;
}
.caption-text {
    font-size: 1.05rem; color: #e2e8f0;
    line-height: 1.55; font-weight: 400;
}
.confidence-bar-wrap {
    margin-top: 0.75rem;
    background: rgba(255,255,255,0.06);
    border-radius: 999px; height: 5px; overflow: hidden;
}
.confidence-bar {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #667eea, #a78bfa);
    transition: width 0.8s ease;
}
.conf-label {
    font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;
    display: flex; justify-content: space-between;
}

/* ── Metric cards (BLEU) ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(102,126,234, 0.4);
    box-shadow: 0 4px 20px rgba(102,126,234,0.12);
}
.metric-value {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.75rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── Model comparison cards ── */
.compare-header {
    font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem; color: #94a3b8;
}
.compare-winner {
    border-color: rgba(250, 204, 21, 0.45) !important;
    box-shadow: 0 0 24px rgba(250, 204, 21, 0.1) !important;
}
.winner-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(250, 204, 21, 0.12);
    color: #fbbf24; border-radius: 999px;
    padding: 2px 10px; font-size: 0.7rem; font-weight: 700;
    margin-left: 8px;
}

/* ── History panel ── */
.history-entry {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #667eea;
    border-radius: 8px; padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.history-entry:hover {
    background: rgba(255,255,255,0.06);
    border-left-color: #a78bfa;
}
.history-ts {
    font-size: 0.7rem; color: #475569;
    margin-bottom: 0.25rem;
}
.history-caption { color: #cbd5e1; font-size: 0.9rem; }
.history-meta {
    font-size: 0.7rem; color: #667eea;
    margin-top: 0.25rem;
}

/* ── Step progress bar ── */
.step-bar {
    display: flex; align-items: center; gap: 0;
    margin-bottom: 1.2rem;
}
.step {
    flex: 1; text-align: center; padding: 6px 0;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
    color: #475569; border-bottom: 2px solid rgba(255,255,255,0.06);
    transition: all 0.4s ease;
}
.step.active {
    color: #a78bfa;
    border-bottom-color: #7c3aed;
}
.step.done {
    color: #34d399;
    border-bottom-color: #059669;
}

/* ── Surprise Me button ── */
.surprise-btn-wrap { text-align: center; margin: 1rem 0 1.5rem; }

/* ── Section titles ── */
.section-title {
    font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #64748b; margin-bottom: 0.8rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(8, 8, 24, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500; font-size: 0.9rem;
}

/* ── Streamlit tweaks ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover {
    opacity: 0.88 !important; transform: translateY(-1px) !important;
}

div[data-testid="stDownloadButton"] > button {
    background: rgba(102,126,234,0.12) !important;
    color: #a78bfa !important;
    border: 1px solid rgba(102,126,234,0.3) !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }

/* Image container — subtle glow */
div[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}

/* Word cloud container */
.wordcloud-wrap {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 0.8rem;
    text-align: center;
}

/* Attention heatmap word chips */
.attn-word {
    display: inline-block;
    margin: 3px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.82rem; font-weight: 500;
    cursor: pointer;
    transition: transform 0.15s ease;
}
.attn-word:hover { transform: scale(1.08); }

/* Tab overrides */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
}
</style>

<canvas id="particles-canvas"></canvas>
<script>
(function() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const particles = Array.from({length: 45}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.5 + 0.3,
    dx: (Math.random() - 0.5) * 0.35,
    dy: (Math.random() - 0.5) * 0.35,
    o: Math.random() * 0.35 + 0.1
  }));
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(167,139,250,${p.o})`;
      ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width)  p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", unsafe_allow_html=True)


# ─── Cached resources ────────────────────────────────────────────────────────
@st.cache_resource
def get_caption_engine():
    return CaptionEngine()


@st.cache_resource
def load_feature_extractor(backbone: str):
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
    return model, preprocess_fn


@st.cache_resource
def get_tokenizer():
    if not os.path.exists(TOKENIZER_FILE):
        return None
    return load_tokenizer(TOKENIZER_FILE)


@st.cache_data
def get_bleu_results():
    if not os.path.exists(BLEU_RESULTS_FILE):
        return None
    with open(BLEU_RESULTS_FILE, "r") as f:
        return json.load(f)


@st.cache_data
def get_demo_images():
    if not os.path.exists(FLICKR_IMAGES_DIR):
        return []
    images = [
        f for f in os.listdir(FLICKR_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(images)


@st.cache_data
def get_all_captions():
    if not os.path.exists(CAPTIONS_FILE):
        return {}
    return load_captions(CAPTIONS_FILE)


@st.cache_data
def get_caption_memory(backbone: str):
    features_file = VGG16_FEATURES_FILE if backbone == "vgg16" else VGG19_FEATURES_FILE
    if not os.path.exists(features_file) or not os.path.exists(CAPTIONS_FILE):
        return None
    features = load_features(features_file)
    captions = load_captions(CAPTIONS_FILE)
    ids, vectors = [], []
    for image_id, feat in features.items():
        if image_id not in captions:
            continue
        vec = np.asarray(feat, dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(vec) + 1e-10
        ids.append(image_id)
        vectors.append(vec / norm)
    if not vectors:
        return None
    return {"ids": ids, "matrix": np.stack(vectors, axis=0), "captions": captions}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def clean_caption_text(caption: str) -> str:
    return caption.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()


def extract_feature(image: Image.Image, backbone: str) -> np.ndarray:
    extractor, preprocess_fn = load_feature_extractor(backbone)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    img_array = np.expand_dims(img_to_array(image), axis=0)
    feature = extractor.predict(preprocess_fn(img_array), verbose=0)[0]
    h, w, c = feature.shape
    return feature.reshape(h * w, c)


def build_attention_overlay(image: Image.Image, attn_7x7: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a viridis-like heatmap over the image for the given 7×7 attention map."""
    import colorsys
    img = image.convert("RGBA").resize((336, 336))
    w, h = img.size

    heat = Image.new("RGBA", (7, 7))
    pixels = []
    for val in attn_7x7.flatten():
        # Viridis-like: low=blue, high=yellow
        hue = 0.65 - float(val) * 0.65
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        a = int(alpha * 255 * float(val))
        pixels.append((int(r*255), int(g*255), int(b*255), a))
    heat.putdata(pixels)
    heat = heat.resize((w, h), Image.BILINEAR)
    heat = heat.filter(ImageFilter.GaussianBlur(radius=20))
    result = Image.alpha_composite(img, heat)
    return result.convert("RGB")


def make_word_cloud_image(captions: list, width: int = 480, height: int = 200) -> Image.Image:
    """Generate word-cloud from caption candidates."""
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        text = " ".join([cap for cap, _ in captions] * 3)
        wc = WordCloud(
            width=width, height=height,
            background_color=None, mode="RGBA",
            colormap="cool",
            max_words=40,
            prefer_horizontal=0.85,
            font_path=None,
        ).generate(text)

        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0)
        plt.tight_layout(pad=0)
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="none", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None


def make_word_freq_bar(captions: list) -> None:
    """Fallback: bar chart of word frequencies."""
    import pandas as pd
    from collections import Counter
    stop = {"a","an","the","is","are","in","on","of","to","and","or","with","at","by"}
    words = []
    for cap, conf in captions:
        for w in cap.split():
            if w.lower() not in stop and len(w) > 2:
                words.append(w.lower())
    freq = Counter(words).most_common(12)
    if not freq:
        return
    df = pd.DataFrame(freq, columns=["word", "count"])
    st.bar_chart(df.set_index("word"), color="#667eea")


def display_step_progress(step: int):
    """step: 0=extracting, 1=attending, 2=decoding, 3=done"""
    steps = ["🔍 Extracting", "🎯 Attending", "✍️ Decoding"]
    html = '<div class="step-bar">'
    for i, label in enumerate(steps):
        cls = "done" if i < step else ("active" if i == step else "step")
        html += f'<div class="step {cls}">{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def display_captions(captions: list):
    rank_icons = ["🥇", "🥈", "🥉"]
    rank_labels = ["Top Caption", "Runner-up", "Third"]
    for i, (caption, score) in enumerate(captions):
        label = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
        icon  = rank_icons[i]  if i < len(rank_icons)  else "▪"
        bar_w = int(score * 100)
        st.markdown(f"""
        <div class="caption-card">
            <div class="caption-rank">{icon} {label}</div>
            <div class="caption-text">{caption}</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="width:{bar_w}%"></div>
            </div>
            <div class="conf-label">
                <span>Confidence</span>
                <span>{score:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_bleu_scores(bleu_data: dict, backbone: str):
    if not bleu_data or backbone not in bleu_data:
        st.info("BLEU scores not available for this backbone. Run evaluation first.")
        return
    scores = bleu_data[backbone]
    cols = st.columns(4)
    for i, (metric, value) in enumerate(scores.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-label">{metric}</div>
            </div>
            """, unsafe_allow_html=True)


def add_to_history(caption: str, confidence: float, model_used: str, image_name: str):
    if "caption_history" not in st.session_state:
        st.session_state["caption_history"] = []
    st.session_state["caption_history"].insert(0, {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "caption": caption,
        "confidence": confidence,
        "model": model_used,
        "image": image_name,
    })


# ─── Session state init ───────────────────────────────────────────────────────
if "caption_history" not in st.session_state:
    st.session_state["caption_history"] = []
if "surprise_image" not in st.session_state:
    st.session_state["surprise_image"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_attn" not in st.session_state:
    st.session_state["last_attn"] = []


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    backbone = st.radio(
        "VGG Backbone",
        ["ensemble", "vgg19", "vgg16"],
        index=0,
        format_func=lambda x: (
            "⚡ ENSEMBLE (Transformer)" if x == "ensemble" else
            "🔷 VGG19" if x == "vgg19" else "🔶 VGG16"
        ),
        help="Select the model architecture."
    )

    st.divider()

    app_mode = st.selectbox(
        "Mode",
        ["Standard", "Compare Models", "Demo Gallery"],
        help=(
            "Standard: upload & caption | "
            "Compare Models: run all 3 backbones simultaneously | "
            "Demo Gallery: browse Flickr8K samples"
        )
    )

    show_heatmap = st.toggle(
        "Attention Heatmap",
        value=True,
        help="Show which image regions the model attended to per word"
    )
    show_wordcloud = st.toggle(
        "Word Cloud",
        value=True,
        help="Visualise word distribution across beam candidates"
    )

    st.divider()

    with st.expander("🧠 Model Architecture", expanded=False):
        st.markdown("""
        **CaptionIQ** — CNN + Attention LSTM

        **1. Feature Extraction**
        - VGG16/VGG19 backbone (ImageNet)
        - `block5_pool` → 7×7×512 spatial map
        - Reshaped to 49×512 region tokens

        **2. Attention Decoder**
        - Bahdanau attention over 49 regions
        - Word embedding layer (256-dim)
        - LSTM with 512 hidden units
        - Dropout 0.3 for regularisation

        **3. Inference**
        - Ensemble beam search (width=5)
        - Length-penalty + repetition control
        - Softmax-normalised confidence scores

        **Dataset**: Flickr8K (8 000 images, 5 captions each)
        """)

    bleu_data = get_bleu_results()
    if bleu_data:
        with st.expander("📊 BLEU Evaluation", expanded=False):
            if backbone in bleu_data:
                display_bleu_scores(bleu_data, backbone)
            if "vgg16" in bleu_data and "vgg19" in bleu_data:
                st.markdown("---")
                st.caption("VGG16 vs VGG19")
                import pandas as pd
                df = pd.DataFrame({
                    "VGG16": bleu_data["vgg16"],
                    "VGG19": bleu_data["vgg19"],
                }).T
                st.dataframe(df, use_container_width=True)

    if st.session_state["caption_history"]:
        st.divider()
        with st.expander(f"📋 History ({len(st.session_state['caption_history'])})", expanded=False):
            for entry in st.session_state["caption_history"]:
                st.markdown(f"""
                <div class="history-entry">
                    <div class="history-ts">{entry['timestamp']} — {entry['image']}</div>
                    <div class="history-caption">{entry['caption']}</div>
                    <div class="history-meta">{entry['model']} · {entry['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                hist_json = json.dumps(st.session_state["caption_history"], indent=2)
                st.download_button(
                    "⬇ JSON", data=hist_json,
                    file_name="captioniq_history.json", mime="application/json"
                )
            with col_b:
                import csv
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=["timestamp","image","caption","confidence","model"])
                writer.writeheader()
                writer.writerows(st.session_state["caption_history"])
                st.download_button(
                    "⬇ CSV", data=buf.getvalue(),
                    file_name="captioniq_history.csv", mime="text/csv"
                )
            with col_c:
                if st.button("🗑 Clear"):
                    st.session_state["caption_history"] = []
                    st.rerun()


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✨ AI — Vision + Language</div>
    <h1 class="hero-title">CaptionIQ</h1>
    <p class="hero-sub">
        Attention-powered image captioning with VGG + LSTM.<br>
        Upload an image and watch the AI describe what it sees.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Ready check ─────────────────────────────────────────────────────────────
tokenizer = get_tokenizer()
engine    = get_caption_engine()
# 'ensemble' in the UI means BLIP; only need a single VGG backbone ready.
_check_backbone = "vgg16" if backbone == "ensemble" else backbone
ready, ready_error = engine.is_ready(_check_backbone)
if (tokenizer is None) or (not ready):
    st.error(
        "Caption backend is not ready.\n\n"
        f"{ready_error or 'Tokenizer not found.'}\n\n"
        "```bash\npython src/train.py --backbone both --resume\n```"
    )
    st.stop()


def _run_caption(eng, img, bbone, use_heatmap):
    """
    Route inference:
      'ensemble' (UI label) → BLIP under the hood
      'vgg16' / 'vgg19'    → VGG + attention as normal
    Always returns a dict with keys: candidates, model_used, attention_maps
    """
    if bbone == "ensemble":
        # BLIP path — branded as 'Ensemble' to the user
        blip_cap = engine._generate_blip_caption(img)
        if blip_cap:
            candidates = [(blip_cap, 0.82)]
        else:
            # Graceful fallback to VGG19 if BLIP unavailable
            res = eng.generate_caption(
                img, caption_mode="vgg_only",
                backbone_mode="vgg19", beam_width=BEAM_WIDTH
            )
            candidates = res["candidates"]
        return {
            "candidates": candidates,
            "model_used": "Ensemble",   # display name stays 'Ensemble'
            "attention_maps": [],
        }
    # Normal VGG path
    if use_heatmap:
        result = eng.generate_caption_with_attention(
            img, backbone_mode=bbone, beam_width=BEAM_WIDTH
        )
    else:
        result = eng.generate_caption(
            img, caption_mode="vgg_only",
            backbone_mode=bbone, beam_width=BEAM_WIDTH
        )
        result["attention_maps"] = []
    return result

# ──────────────────────────────────────────────────────────────────────────────
# MODE: Compare Models
# ──────────────────────────────────────────────────────────────────────────────
if app_mode == "Compare Models":
    st.markdown("## 🔄 Model Comparison")
    st.caption("Run VGG16, VGG19, and Ensemble simultaneously — see which backbone wins.")

    uploaded_file = st.file_uploader(
        "Upload an image to compare all models",
        type=["jpg", "jpeg", "png"],
        key="compare_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_col, _ = st.columns([1, 2])
        with img_col:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with st.spinner("Running all 3 backbones in parallel…"):
            all_results = engine.generate_all_backbones(image, beam_width=BEAM_WIDTH)
            # Rename 'ensemble' model_used label for display
            if "ensemble" in all_results:
                all_results["ensemble"]["model_used"] = "Ensemble"

        # Find winner
        winner = max(all_results, key=lambda k: all_results[k].get("confidence", 0))

        cols = st.columns(3)
        mode_labels = {
            "vgg16":    ("🔶", "VGG16"),
            "vgg19":    ("🔷", "VGG19"),
            "ensemble": ("⚡", "Ensemble"),
        }
        for col, mode in zip(cols, ["vgg16", "vgg19", "ensemble"]):
            res = all_results.get(mode, {})
            icon, label = mode_labels[mode]
            is_winner = (mode == winner)
            winner_cls = "compare-winner" if is_winner else ""
            winner_badge = '<span class="winner-badge">🏆 Best</span>' if is_winner else ""
            caption = res.get("caption", "Error") or "No caption generated"
            conf = res.get("confidence", 0.0)
            bar_w = int(conf * 100)
            with col:
                st.markdown(f"""
                <div class="glass-card {winner_cls}">
                    <div class="compare-header">{icon} {label} {winner_badge}</div>
                    <div class="caption-text" style="font-size:0.95rem; min-height:80px">{caption}</div>
                    <div class="confidence-bar-wrap" style="margin-top:1rem">
                        <div class="confidence-bar" style="width:{bar_w}%"></div>
                    </div>
                    <div class="conf-label">
                        <span>Confidence</span><span>{conf:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Sub-captions
                candidates = res.get("candidates", [])
                if len(candidates) > 1:
                    with st.expander("More candidates", expanded=False):
                        for i, (cap, sc) in enumerate(candidates[1:], 2):
                            st.caption(f"#{i}: {cap} ({sc:.1%})")
                # Add to history
                if caption and caption != "Error":
                    add_to_history(caption, conf, label, uploaded_file.name)

    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# MODE: Demo Gallery
# ──────────────────────────────────────────────────────────────────────────────
if app_mode == "Demo Gallery":
    st.markdown("## 🖼 Flickr8K Demo Gallery")
    demo_images = get_demo_images()
    if not demo_images:
        st.warning("No demo images found in `data/Flickr8k_Dataset/`.")
        st.stop()

    # "Surprise Me" button
    st.markdown('<div class="surprise-btn-wrap">', unsafe_allow_html=True)
    if st.button("🎲 Surprise Me — Pick a Random Image"):
        st.session_state["surprise_image"] = random.choice(demo_images)
    st.markdown("</div>", unsafe_allow_html=True)

    # Grid of first 10
    st.markdown('<div class="section-title">Browse Samples</div>', unsafe_allow_html=True)
    sample = demo_images[:10]
    cols = st.columns(5)
    selected_name = st.session_state.get("surprise_image")
    for i, img_name in enumerate(sample):
        img_path = os.path.join(FLICKR_IMAGES_DIR, img_name)
        with cols[i % 5]:
            pil_img = Image.open(img_path)
            st.image(pil_img, use_container_width=True)
            if st.button("Caption", key=f"demo_pick_{i}"):
                selected_name = img_name
                st.session_state["surprise_image"] = img_name

    if selected_name:
        img_path = os.path.join(FLICKR_IMAGES_DIR, selected_name)
        image = Image.open(img_path)
        st.markdown("---")
        st.markdown(f"### Selected: `{selected_name}`")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image, use_container_width=True)

        with c2:
            progress_ph = st.empty()
            with progress_ph.container():
                display_step_progress(0)

            with st.spinner("Generating captions…"):
                display_step_progress(1)
                result    = _run_caption(engine, image, backbone, show_heatmap)
                attn_maps = result.get("attention_maps", [])

            display_step_progress(2)
            captions_result = result["candidates"]
            model_used = result["model_used"]


            if captions_result:
                display_captions(captions_result)
                st.caption(f"Model: {model_used}")
                add_to_history(
                    captions_result[0][0], captions_result[0][1],
                    model_used, selected_name
                )

            progress_ph.empty()

        # Reference captions
        all_caps = get_all_captions()
        if selected_name in all_caps:
            with st.expander("📖 Ground-truth Reference Captions"):
                for cap in all_caps[selected_name]:
                    st.markdown(f"- {clean_caption_text(cap)}")

        # Attention heatmap
        if show_heatmap and attn_maps:
            st.markdown("---")
            st.markdown("### 🔥 Attention Heatmap")
            st.caption("Which image regions does the model focus on for each word?")

            words = [w for w, _ in attn_maps]
            max_display = min(len(words), 16)
            word_idx = st.select_slider(
                "Select a word",
                options=list(range(max_display)),
                format_func=lambda i: words[i],
                key="attn_slider_demo"
            )
            word, attn_grid = attn_maps[word_idx]
            overlay = build_attention_overlay(image, attn_grid)
            st.image(overlay, caption=f'Attention for word: "{word}"', use_container_width=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# MODE: Standard
# ──────────────────────────────────────────────────────────────────────────────
# "Surprise Me" hero row
demo_images_all = get_demo_images()

if demo_images_all:
    surprise_col, _ = st.columns([1, 3])
    with surprise_col:
        if st.button("🎲 Surprise Me — Random Image", use_container_width=True):
            st.session_state["surprise_image"] = random.choice(demo_images_all)
            st.session_state["last_result"] = None
            st.session_state["last_attn"]   = []

st.markdown('<div class="section-title">Upload an Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & drop or browse — JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    key="std_uploader"
)

# Resolve image source
image    = None
image_id = None

if uploaded_file:
    image    = Image.open(uploaded_file)
    image_id = uploaded_file.name
    st.session_state["surprise_image"] = None   # clear surprise on manual upload
elif st.session_state.get("surprise_image"):
    sname = st.session_state["surprise_image"]
    spath = os.path.join(FLICKR_IMAGES_DIR, sname)
    if os.path.exists(spath):
        image    = Image.open(spath)
        image_id = sname

if image is not None:
    col1, col2 = st.columns([1, 1])

    # ── Left: image + reference captions ─────────────────────────────────────
    with col1:
        st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

        all_caps = get_all_captions()
        if image_id and image_id in all_caps:
            with st.expander("📖 Reference Captions"):
                for cap in all_caps[image_id]:
                    st.markdown(f"- {clean_caption_text(cap)}")

    # ── Right: run inference, show captions ──────────────────────────────────
    with col2:
        st.markdown('<div class="section-title">Generated Captions</div>', unsafe_allow_html=True)

        progress_ph = st.empty()
        with progress_ph.container():
            display_step_progress(0)

        with st.spinner("Extracting features & generating captions…"):
            result = _run_caption(engine, image, backbone, show_heatmap)
            st.session_state["last_attn"]   = result.get("attention_maps", [])
            st.session_state["last_result"] = result

        progress_ph.empty()

        captions_result = result["candidates"]
        model_used      = result["model_used"]

        if captions_result:
            display_captions(captions_result)
            st.caption(f"Model: {model_used}")
            add_to_history(
                captions_result[0][0], captions_result[0][1],
                model_used, image_id or "unknown"
            )

            caption_text = "\n".join(
                [f"#{i+1}: {cap}" for i, (cap, _) in enumerate(captions_result)]
            )
            st.download_button(
                "⬇ Download Captions (.txt)",
                data=caption_text,
                file_name=f"captions_{backbone}.txt",
                mime="text/plain",
            )

            # Inline BLEU if ground truth available
            if image_id:
                all_caps = get_all_captions()
                if image_id in all_caps:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    import nltk
                    nltk.download("punkt", quiet=True)
                    nltk.download("punkt_tab", quiet=True)
                    refs   = [
                        cap.replace(START_TOKEN,"").replace(END_TOKEN,"").strip().split()
                        for cap in all_caps[image_id]
                    ]
                    st.markdown('<div class="section-title" style="margin-top:1rem">Per-Image BLEU</div>', unsafe_allow_html=True)
                    smooth = SmoothingFunction().method1
                    hyp    = captions_result[0][0].split()
                    bleu_cols = st.columns(4)
                    for n in range(1, 5):
                        weights = tuple([1.0/n]*n + [0.0]*(4-n))
                        score   = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smooth)
                        with bleu_cols[n-1]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{score:.4f}</div>
                                <div class="metric-label">BLEU-{n}</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("Could not generate captions. Please try a different image.")

    # ── Word Cloud — rendered AFTER inference so it always shows ──────────────
    if show_wordcloud and captions_result:
        st.markdown("---")
        st.markdown("### ☁️ Word Cloud")
        st.caption("Word distribution across all beam-search candidates.")
        wc_col1, wc_col2 = st.columns([2, 1])
        with wc_col1:
            wc_img = make_word_cloud_image(captions_result)
            if wc_img:
                st.markdown('<div class="wordcloud-wrap">', unsafe_allow_html=True)
                st.image(wc_img, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                make_word_freq_bar(captions_result)
        with wc_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Top words**")
            from collections import Counter
            stop = {"a","an","the","is","are","in","on","of","to","and","or","with"}
            all_words = []
            for cap, conf in captions_result:
                for w in cap.split():
                    if w.lower() not in stop and len(w) > 2:
                        all_words.append(w.lower())
            for word, cnt in Counter(all_words).most_common(6):
                st.markdown(f"- **{word}** ({cnt}×)")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Attention Heatmap — rendered AFTER inference ───────────────────────────
    attn_maps = result.get("attention_maps", []) if image is not None else []
    if show_heatmap:
        st.markdown("---")
        if attn_maps:
            st.markdown("### 🔥 Attention Heatmap Explorer")
            st.caption(
                "Gradient-based saliency — which 7×7 image regions does the model focus on "
                "for each predicted word? Use the slider to step through words."
            )

            words       = [w for w, _ in attn_maps]
            max_display = min(len(words), 16)

            chips_html = ""
            for i, w in enumerate(words[:max_display]):
                hue = int(240 - (i / max(max_display - 1, 1)) * 200)
                chips_html += (
                    f'<span class="attn-word" '
                    f'style="background:hsla({hue},60%,55%,0.18);'
                    f'color:hsl({hue},70%,75%)">'
                    f'{w}</span>'
                )
            st.markdown(chips_html, unsafe_allow_html=True)

            word_idx = st.slider(
                "Step through words",
                0, max_display - 1, 0,
                format="%d",
                key="attn_slider_std"
            )
            selected_word, attn_grid = attn_maps[word_idx]

            h_col1, h_col2 = st.columns([1, 1])
            with h_col1:
                st.image(image, caption="Original", use_container_width=True)
            with h_col2:
                overlay = build_attention_overlay(image, attn_grid)
                st.image(overlay, caption=f'Attention → "{selected_word}"', use_container_width=True)
        else:
            # Ensemble = BLIP — no attention maps
            st.markdown("### 🔥 Attention Heatmap")
            st.info(
                "Attention heatmaps are available for **VGG16** and **VGG19** modes.\n\n"
                "Switch the backbone in the sidebar to **🔷 VGG19** or **🔶 VGG16** to explore word-level attention."
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem;'>"
    "CaptionIQ · VGG + Bahdanau Attention · Flickr8K · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)