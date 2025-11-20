#!/usr/bin/env python3
"""
Login to Hugging Face Hub
"""

from huggingface_hub import login
import os

def hf_login():
    print("🔐 HUGGING FACE LOGIN")
    print("=" * 40)
    
    # Check for token in environment
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ No HF_TOKEN found in environment")
        print("💡 Please set your token:")
        print("   export HF_TOKEN=your_token_here")
        return False
    
    try:
        login(token=token)
        print("✅ Successfully logged in to Hugging Face Hub")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

if __name__ == "__main__":
    hf_login()
