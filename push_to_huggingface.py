#!/usr/bin/env python3
"""
Push Visual Narrator VLM artifacts to an existing Hugging Face repo.
- Uses env token: HUGGINGFACE_HUB_TOKEN (preferred) or HF_TOKEN (fallback)
- Uploads only relevant files from MODEL_PATH
- Updates/creates the model card
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, create_repo

MODEL_PATH = Path("outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982")
REPO_NAME = "visual-narrator-llm"
USERNAME = "Ytgetahun"
FULL_REPO_ID = f"{USERNAME}/{REPO_NAME}"

# Upload only these patterns from the checkpoint folder
ALLOW_PATTERNS = [
    "*.safetensors", "*.bin", "*.pt",
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "merges.txt", "vocab.json",
    "*.md", "*.txt"
]

def get_hf_token() -> str:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("No token found. Set one of the following environment variables:")
        print("  export HUGGINGFACE_HUB_TOKEN=hf_xxx")
        print("  # or")
        print("  export HF_TOKEN=hf_xxx")
        sys.exit(1)
    return token

def build_model_card(repo_id: str) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return """---
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-text
base_model: Salesforce/blip-image-captioning-large
tags:
- vision
- image-captioning
- blip
- adjectives
- descriptive
- visual-narrator
- multimodal
- phase7-complete
---

# Visual Narrator VLM

## Adjective-Dominant Visual Language Model

Transform visual input into richer, more descriptive captions via adjective-dominant narration.

## Phase 7 Highlights

- Avg Adjectives: 5.40 per caption (target >=3)
- Peak: 7 adjectives in a single caption
- Consistency: 100 percent of captions have 3 or more adjectives
- Inference: about 400 ms per image on GPU

## Usage

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

repo_id = "{repo_id}"
processor = BlipProcessor.from_pretrained(repo_id)
model = BlipForConditionalGeneration.from_pretrained(repo_id).to("cuda")

image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.amp.autocast("cuda", enabled=True):
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )

caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
Technical Specs
Base: BLIP image-captioning model

Training: 10,000 steps across 50 epochs (Phase 7)

Dataset: around 3,138 adjective-augmented COCO samples

Precision: FP16 with gradient scaling

Infra: NVIDIA GH200 480GB, PyTorch 2.5.x, Transformers

Benchmark Snapshot
Category	Pass Rate
Nature	100%
Urban	Needs work
People	~50%

Updated: {now}
""".format(repo_id=repo_id, now=now)

def main():
    print("PUSHING VISUAL NARRATOR VLM TO HUGGING FACE")
    print("=" * 60)
    print("PUSHING VISUAL NARRATOR VLM TO HUGGING FACE")
    print("=" * 60)
    
    if not MODEL_PATH.exists() or not MODEL_PATH.is_dir():
        print(f"Model path not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"Model found: {MODEL_PATH}")
    
    token = get_hf_token()
    api = HfApi(token=token)
    
    # Ensure repo exists (no error if it already exists)
    try:
        create_repo(repo_id=FULL_REPO_ID, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Repo check: {e}")
    
    print(f"Target repository: {FULL_REPO_ID}")
    print(f"URL: https://huggingface.co/{FULL_REPO_ID}")
    
    # Upload model files
    print("Uploading model files (filtered)...")
    try:
        api.upload_folder(
            folder_path=str(MODEL_PATH),
            repo_id=FULL_REPO_ID,
            repo_type="model",
            commit_message=f"Phase 7 upload - {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            allow_patterns=ALLOW_PATTERNS
        )
        print("Model files uploaded.")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)
    
    # Update model card
    print("Updating model card...")
    try:
        card_text = build_model_card(FULL_REPO_ID)
        card = ModelCard(card_text)
        card.push_to_hub(FULL_REPO_ID, token=token)
        print("Model card updated.")
    except Exception as e:
        print(f"Model card update failed: {e}")
        sys.exit(1)
    
    print(f"SUCCESS: https://huggingface.co/{FULL_REPO_ID}")
if __name__ == "__main__":
    # Temporarily disable offline mode if set
    original_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if original_offline:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        main()
    finally:
        if original_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = original_offline
