#!/usr/bin/env python3
"""
Simple script to upload model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo

def simple_upload():
print("🚀 SIMPLE UPLOAD TO HUGGING FACE")
print("=" * 50)

text
MODEL_PATH = "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982"
REPO_NAME = "visual-narrator-vlm"
USERNAME = "Ytgetahun"

full_repo_name = f"{USERNAME}/{REPO_NAME}"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    return False

try:
    # Create repo
    create_repo(repo_id=full_repo_name, exist_ok=True, private=False)
    
    # Upload files
    api = HfApi()
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=full_repo_name,
        commit_message="Visual Narrator VLM - Initial upload"
    )
    
    print(f"✅ Model uploaded to: https://huggingface.co/{full_repo_name}")
    print("📝 You can now add the model card manually in the web interface")
    return True
    
except Exception as e:
    print(f"❌ Upload failed: {e}")
    return False
if name == "main":
simple_upload()
