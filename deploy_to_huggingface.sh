#!/bin/bash

# Visual Narrator LLM - Hugging Face Deployment Script
echo "🚀 Deploying Visual Narrator LLM to Hugging Face..."

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Create repository (if it doesn't exist)
REPO_NAME="visual-narrator-opt-2.7b"
echo "📦 Creating repository: $REPO_NAME"

# Login to Hugging Face (if not already logged in)
huggingface-cli login

# Create directory structure
mkdir -p $REPO_NAME
mkdir -p $REPO_NAME/training_phases

# Copy model files
echo "📁 Copying model files..."
cp -r ./phase5/focused_model/* $REPO_NAME/

# Copy documentation
echo "📄 Copying documentation..."
cp *.md $REPO_NAME/training_phases/
cp test_3b_models.py $REPO_NAME/training_phases/

# Copy configuration files
cp huggingface_repo/README.md $REPO_NAME/
cp huggingface_repo/adapter_config.json $REPO_NAME/
cp huggingface_repo/requirements.txt $REPO_NAME/

# Create training summary
cat > $REPO_NAME/training_summary.json << 'EOL'
{
  "model_name": "visual-narrator-opt-2.7b",
  "base_model": "facebook/opt-2.7b",
  "fine_tuning_method": "LoRA",
  "training_phases": {
    "phase_4.1": "2.7B model foundation",
    "phase_4.2": "Dataset integration", 
    "phase_4.3": "Advanced training",
    "phase_4.4": "Benchmarking",
    "phase_5": "SOTA optimization"
  },
  "performance_metrics": {
    "overall_success_rate": 0.88,
    "sota_benchmark_score": 0.577,
    "urban_scene_success": 0.75,
    "people_scene_success": 1.0,
    "training_efficiency": 0.003
  },
  "training_data": {
    "total_examples": 555,
    "focus_areas": ["urban_scenes", "people_activities", "nature_scenes"],
    "data_quality": "professional_curated"
  }
}
EOL

echo "✅ Repository structure created!"

# Upload to Hugging Face
echo "📤 Uploading to Hugging Face Hub..."
cd $REPO_NAME
git init
git add .
git commit -m "Deploy Visual Narrator OPT-2.7B: Fine-tuned for image description with 88% success rate"
git push --set-upstream origin main

echo "🎉 Deployment complete!"
echo "🔗 Your model is available at: https://huggingface.co/your-username/$REPO_NAME"
