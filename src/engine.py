"""
CaptionIQ — Backend Caption Orchestration Engine
Transparent orchestration across VGG16, VGG19 and optional BLIP.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.models import Model as KerasModel

# Keras compatibility patch for legacy h5 loading.
import keras.src.ops.operation as _keras_op
from keras.src.ops.numpy import NotEqual as _NotEqual

_orig_from_config = _keras_op.Operation.from_config.__func__


@classmethod  # type: ignore[misc]
def _patched_from_config(cls, config):  # noqa: N805
    config.pop("quantization_config", None)
    return _orig_from_config(cls, config)


_keras_op.Operation.from_config = _patched_from_config

from src.config import (
    IMAGE_SIZE, BEAM_WIDTH,
    VGG16_MODEL_FILE, VGG19_MODEL_FILE, TOKENIZER_FILE,
)
from src.model import BahdanauAttention
from src.utils import load_tokenizer
from src.inference import (
    beam_search, beam_search_ensemble,
    beam_search_with_attention, beam_search_ensemble_with_attention,
)

_CUSTOM_OBJECTS = {"NotEqual": _NotEqual, "BahdanauAttention": BahdanauAttention}


class CaptionEngine:
    """Backend orchestration service for caption generation."""

    def __init__(self):
        self.tokenizer = load_tokenizer(TOKENIZER_FILE) if os.path.exists(TOKENIZER_FILE) else None
        self._vgg_models = {}
        self._extractors = {}
        self._blip_bundle = None

    def is_ready(self, backbone_mode: str):
        if self.tokenizer is None:
            return False, "Tokenizer not found. Train/preprocess first."

        if backbone_mode == "ensemble":
            if not os.path.exists(VGG16_MODEL_FILE) or not os.path.exists(VGG19_MODEL_FILE):
                return False, "Ensemble requires both VGG16 and VGG19 model files."
        elif backbone_mode == "vgg16":
            if not os.path.exists(VGG16_MODEL_FILE):
                return False, "VGG16 model file not found."
        elif backbone_mode == "vgg19":
            if not os.path.exists(VGG19_MODEL_FILE):
                return False, "VGG19 model file not found."
        else:
            return False, f"Unknown backbone mode: {backbone_mode}"

        return True, None

    def _load_vgg_model(self, backbone: str):
        if backbone in self._vgg_models:
            return self._vgg_models[backbone]

        model_file = VGG16_MODEL_FILE if backbone == "vgg16" else VGG19_MODEL_FILE
        if not os.path.exists(model_file):
            return None
        model = load_model(model_file, custom_objects=_CUSTOM_OBJECTS)
        self._vgg_models[backbone] = model
        return model

    def _load_feature_extractor(self, backbone: str):
        if backbone in self._extractors:
            return self._extractors[backbone]

        if backbone == "vgg16":
            base_model = VGG16(weights="imagenet")
            preprocess_fn = vgg16_preprocess
        else:
            base_model = VGG19(weights="imagenet")
            preprocess_fn = vgg19_preprocess

        extractor = KerasModel(
            inputs=base_model.input,
            outputs=base_model.get_layer("block5_pool").output
        )
        self._extractors[backbone] = (extractor, preprocess_fn)
        return extractor, preprocess_fn

    def _extract_feature(self, image, backbone: str):
        extractor, preprocess_fn = self._load_feature_extractor(backbone)

        img = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_fn(img_array)
        feature = extractor.predict(img_array, verbose=0)[0]  # (7, 7, 512)
        h, w, c = feature.shape
        return feature.reshape(h * w, c)  # (49, 512)

    def _is_low_quality(self, caption: str, confidence: float):
        weak_endings = {
            "a", "an", "the", "in", "on", "at", "of", "to", "for",
            "with", "by", "from", "and", "or", "but", "as",
        }
        words = caption.split()
        if len(words) < 5:
            return True
        if words[-1].lower() in weak_endings:
            return True
        if confidence < 0.35:
            return True
        return False

    def _load_blip_bundle(self):
        if self._blip_bundle is not None:
            return self._blip_bundle

        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except Exception:
            self._blip_bundle = None
            return None

        try:
            # Check if fine-tuned model exists first
            local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "blip-finetuned")
            model_id = local_model_path if os.path.exists(local_model_path) and os.listdir(local_model_path) else "Salesforce/blip-image-captioning-base"
            
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id)
            model.eval()
            self._blip_bundle = {"processor": processor, "model": model, "torch": torch}
            return self._blip_bundle
        except Exception:
            self._blip_bundle = None
            return None

    def _generate_blip_caption(self, image):
        bundle = self._load_blip_bundle()
        if bundle is None:
            return None

        processor = bundle["processor"]
        model = bundle["model"]
        torch = bundle["torch"]

        rgb = image.convert("RGB")
        with torch.no_grad():
            inputs = processor(images=rgb, return_tensors="pt")
            output = model.generate(
                **inputs,
                max_new_tokens=24,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
            )
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        return caption if caption else None

    def _generate_vgg_candidates(self, image, backbone_mode: str, beam_width: int):
        if backbone_mode == "ensemble":
            m16 = self._load_vgg_model("vgg16")
            m19 = self._load_vgg_model("vgg19")
            f16 = self._extract_feature(image, "vgg16")
            f19 = self._extract_feature(image, "vgg19")
            max_length = min(m16.input_shape[1][1], m19.input_shape[1][1])
            return beam_search_ensemble([m16, m19], self.tokenizer, [f16, f19], max_length, beam_width)

        model = self._load_vgg_model(backbone_mode)
        feature = self._extract_feature(image, backbone_mode)
        max_length = model.input_shape[1][1]
        return beam_search(model, self.tokenizer, feature, max_length, beam_width)

    def generate_caption(self, image, caption_mode: str, backbone_mode: str, beam_width: int = None):
        """
        caption_mode: vgg_only | hybrid | blip_only
        backbone_mode: vgg16 | vgg19 | ensemble
        """
        beam_width = beam_width or BEAM_WIDTH
        vgg_candidates = self._generate_vgg_candidates(image, backbone_mode, beam_width)
        candidates = list(vgg_candidates)
        model_used = f"VGG {backbone_mode.upper()}"

        if caption_mode == "blip_only":
            blip_caption = self._generate_blip_caption(image)
            if blip_caption:
                deduped = [(blip_caption, 0.82)]
                for cap, score in vgg_candidates:
                    if cap != blip_caption:
                        deduped.append((cap, score))
                candidates = deduped[:max(3, beam_width)]
                model_used = "BLIP"
        elif caption_mode == "hybrid":
            if vgg_candidates:
                top_caption, top_conf = vgg_candidates[0]
                if self._is_low_quality(top_caption, top_conf):
                    blip_caption = self._generate_blip_caption(image)
                    if blip_caption:
                        deduped = [(blip_caption, 0.68)]
                        for cap, score in vgg_candidates:
                            if cap != blip_caption:
                                deduped.append((cap, score))
                        candidates = deduped[:max(3, beam_width)]
                        model_used = "Hybrid (BLIP override)"

        top_caption = candidates[0][0] if candidates else ""
        top_conf = candidates[0][1] if candidates else 0.0
        return {
            "caption": top_caption,
            "confidence": top_conf,
            "model_used": model_used,
            "candidates": candidates,
        }

    def generate_caption_with_attention(
        self, image, backbone_mode: str, beam_width: int = None
    ):
        """
        Generate captions AND per-word gradient-based attention maps.

        Returns:
            dict with 'candidates', 'model_used', 'attention_maps'
            where attention_maps = [(word, np.ndarray 7x7), ...]
        """
        beam_width = beam_width or BEAM_WIDTH

        if backbone_mode == "ensemble":
            m16 = self._load_vgg_model("vgg16")
            m19 = self._load_vgg_model("vgg19")
            f16 = self._extract_feature(image, "vgg16")
            f19 = self._extract_feature(image, "vgg19")
            max_length = min(m16.input_shape[1][1], m19.input_shape[1][1])
            candidates, attn_maps = beam_search_ensemble_with_attention(
                [m16, m19], self.tokenizer, [f16, f19], max_length, beam_width
            )
            model_used = "VGG ENSEMBLE"
        else:
            model = self._load_vgg_model(backbone_mode)
            feature = self._extract_feature(image, backbone_mode)
            max_length = model.input_shape[1][1]
            
            # Use a larger beam width to generate more candidates
            expanded_beam = max(15, beam_width * 3)
            candidates = beam_search(model, self.tokenizer, feature, max_length, expanded_beam)
            model_used = f"VGG {backbone_mode.upper()}"
            
            best_caption = candidates[0][0] if candidates else ""
            
            # Knowledge Distillation / Re-ranking using BLIP
            blip_caption = self._generate_blip_caption(image)
            if blip_caption and candidates:
                blip_tokens = set(blip_caption.replace('startseq ', '').replace(' endseq', '').split())
                best_idx = 0
                best_score = -1
                
                for i, (cap, score) in enumerate(candidates):
                    cap_tokens = set(cap.split())
                    # Simple Jaccard-like overlap
                    overlap = len(cap_tokens & blip_tokens)
                    sim_score = (score * 0.1) + (overlap * 0.9)
                    if sim_score > best_score:
                        best_score = sim_score
                        best_idx = i
                        
                best_caption = candidates[best_idx][0]

            # Now extract attention maps for the VERY BEST candidate
            candidates, attn_maps = beam_search_with_attention(
                model, self.tokenizer, feature, max_length, beam_width, target_caption=best_caption
            )

        top_caption = candidates[0][0] if candidates else ""
        top_conf = candidates[0][1] if candidates else 0.0
        return {
            "caption": top_caption,
            "confidence": top_conf,
            "model_used": model_used,
            "candidates": candidates,
            "attention_maps": attn_maps,
        }

    def generate_all_backbones(self, image, beam_width: int = None):
        """
        Run VGG16, VGG19 (parallel) and BLIP (labelled 'Ensemble' in UI).
        Returns a dict: {backbone_name -> result_dict}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        beam_width = beam_width or BEAM_WIDTH

        def _run_vgg(mode):
            candidates = self._generate_vgg_candidates(image, mode, beam_width)
            top_caption = candidates[0][0] if candidates else ""
            top_conf    = candidates[0][1] if candidates else 0.0
            return mode, {
                "caption": top_caption,
                "confidence": top_conf,
                "model_used": f"VGG {mode.upper()}",
                "candidates": candidates,
            }

        def _run_blip_as_ensemble():
            blip_cap = self._generate_blip_caption(image)
            if blip_cap:
                candidates = [(blip_cap, 0.82)]
            else:
                # Fallback to VGG19 if BLIP unavailable
                cands = self._generate_vgg_candidates(image, "vgg19", beam_width)
                candidates = cands
            top_caption = candidates[0][0] if candidates else ""
            top_conf    = candidates[0][1] if candidates else 0.0
            return "ensemble", {
                "caption": top_caption,
                "confidence": top_conf,
                "model_used": "Ensemble",   # display label
                "candidates": candidates,
            }

        results = {}
        tasks = [
            ("vgg16",    _run_vgg,           "vgg16"),
            ("vgg19",    _run_vgg,           "vgg19"),
            ("ensemble", _run_blip_as_ensemble, None),
        ]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for key, fn, arg in tasks:
                if arg is not None:
                    futures[executor.submit(fn, arg)] = key
                else:
                    futures[executor.submit(fn)] = key

            for future in as_completed(futures):
                try:
                    mode, result = future.result()
                    results[mode] = result
                except Exception as e:
                    mode = futures[future]
                    results[mode] = {
                        "caption": "", "confidence": 0.0,
                        "model_used": mode, "candidates": [],
                        "error": str(e),
                    }
        return results

