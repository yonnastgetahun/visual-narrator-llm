#!/usr/bin/env python3
"""
Complete script to upload Visual Narrator VLM to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from datetime import datetime

def upload_model():
    print("🚀 UPLOADING VISUAL NARRATOR VLM TO HUGGING FACE")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982"
    REPO_NAME = "visual-narrator-vlm"
    USERNAME = "Ytgetahun"
    
    full_repo_name = f"{USERNAME}/{REPO_NAME}"
    
    # Verify model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    
    print(f"✅ Model found: {MODEL_PATH}")
    print(f"📦 Target repository: {full_repo_name}")
    
    try:
        # Create repository
        print("🔄 Creating repository...")
        create_repo(repo_id=full_repo_name, exist_ok=True, private=False)
        
        # Initialize HF API
        api = HfApi()
        
        # Upload model
        print("📤 Uploading model files...")
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=full_repo_name,
            commit_message=f"Visual Narrator VLM v1.0 - {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        print(f"✅ SUCCESS: Model uploaded to https://huggingface.co/{full_repo_name}")
        print("🎉 Visual Narrator VLM is now on Hugging Face Hub!")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_model()
