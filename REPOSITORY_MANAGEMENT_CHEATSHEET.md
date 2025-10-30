# 🔄 Repository Management Cheatsheet

## Daily Workflow:
```bash
# 1. Sync from Hugging Face (primary)
git pull --rebase origin main

# 2. Make changes and commit
git add .
git commit -m "description"

# 3. Push to both repositories
git push origin main    # HF: full project
git push github main    # GitHub: code-only 
# Hugging Face (full project):
origin    https://huggingface.co/Ytgetahun/visual-narrator-llm

# GitHub (code-only):
github    https://github.com/yonnastgetahun/visual-narrator-llm.git 
cat > setup_remotes.sh << 'EOF'
#!/bin/bash
echo "🔧 Setting up Git remotes for Visual Narrator LLM" 
