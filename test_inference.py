import sys
import os
from PIL import Image
from src.engine import CaptionEngine
from src.config import FLICKR_IMAGES_DIR, TEST_IMAGES_FILE
from src.utils import load_image_list

print("Loading engine...")
engine = CaptionEngine()

test_images = load_image_list(TEST_IMAGES_FILE)
test_image = sorted(list(test_images))[0]

image_path = os.path.join(FLICKR_IMAGES_DIR, test_image)
image = Image.open(image_path)

print(f"Generating caption for {test_image} using BLIP (Ensemble fallback)...")
res = engine.generate_caption(image, caption_mode="blip_only", backbone_mode="ensemble")
print("Caption:", res["caption"])
print("Model used:", res["model_used"])
