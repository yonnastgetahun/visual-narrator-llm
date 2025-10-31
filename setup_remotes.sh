#!/bin/bash
echo "🔧 Setting up Git remotes for Visual Narrator LLM"

# Remove existing remotes (if any)
git remote remove origin 2>/dev/null
git remote remove github 2>/dev/null

# Add Hugging Face remote (with token)
git remote add origin https://Ytgetahun:HF_TOKEN_REDACTED@huggingface.co/Ytgetahun/visual-narrator-llm

# Add GitHub remote
git remote add github https://github.com/yonnastgetahun/visual-narrator-llm.git

echo "✅ Remotes configured:"
git remote -v

echo ""
echo "🚀 Usage:"
echo "   git push origin main    # Push to Hugging Face"
echo "   git push github main    # Push to GitHub"
