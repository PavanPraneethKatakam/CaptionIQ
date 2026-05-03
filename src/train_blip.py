"""
CaptionIQ — BLIP Fine-Tuning Script
This script fine-tunes the state-of-the-art BLIP vision-language foundation model
on the Flickr8K dataset. It leverages HuggingFace Transformers and PyTorch, 
and is optimized to run on MPS (Mac) or CUDA (T4/Colab).
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ─── Configuration ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "Flickr8k_Dataset")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions_clean.txt")
SAVE_DIR = os.path.join(BASE_DIR, "models", "blip-finetuned")

EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

# ─── Device Selection ─────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Training on CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # Setting high watermark ratio helps prevent MPS OOM errors
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    print("🚀 Training on MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("⚠️ Training on CPU (Will be very slow!)")


# ─── Dataset Class ────────────────────────────────────────────
class Flickr8kDataset(Dataset):
    def __init__(self, captions_file, images_dir, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.data = []

        print(f"Loading captions from {captions_file}...")
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) < 2:
                    continue
                img_id = tokens[0]
                caption = tokens[1]
                
                # Clean 'startseq' and 'endseq' which are specific to the legacy LSTM model
                caption = caption.replace("startseq", "").replace("endseq", "").strip()
                self.data.append({"image_id": img_id, "caption": caption})
                
        print(f"Loaded {len(self.data)} image-caption pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item["image_id"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to a blank image if corrupted
            image = Image.new("RGB", (224, 224), (255, 255, 255))
            
        encoding = self.processor(
            images=image, 
            text=item["caption"], 
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by the processor
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


# ─── Training Loop ────────────────────────────────────────────
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("\nLoading BLIP Processor and Model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    
    dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids, # In BLIP, labels are the input_ids for language modeling
                attention_mask=attention_mask
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        print(f"Saving checkpoint to {SAVE_DIR}...")
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)

    print("\n✅ Training Complete! Model saved successfully.")
    print("You can now update your Streamlit app to load from 'models/blip-finetuned' instead of Salesforce.")

if __name__ == "__main__":
    train()
