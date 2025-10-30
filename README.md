# Visual Narrator LLM 🎥→📝

## Project Goal
Create a state-of-the-art 3B parameter LLM specialized in generating vivid audio descriptions for visual media.

## Current Status
🎯 **Phase 1: Practice & Setup - COMPLETED ✅**
- ✅ Environment setup completed
- ✅ Python 3.12 compatibility verified
- ✅ Data loading pipelines tested and working
- ✅ Conceptual Captions dataset accessible
- 🚀 **READY FOR FIRST TRAINING RUN**

## Recent Achievements
- Solved Python 3.12 package compatibility issues
- Established working development environment
- Successfully tested multiple dataset loaders:
  - ✅ Wikitext (text)
  - ✅ Conceptual Captions (image URLs + captions)
  - ✅ CIFAR-10 (images)

## Immediate Next Step
Running first training script: `training/train_conceptual.py`

## Architecture Plan
- **Base Model**: 3B parameter decoder-only transformer
- **Training**: Pre-training on Common Crawl + specialized fine-tuning
- **Target**: SOTA performance on audio description tasks

## Technical Stack
- PyTorch 2.0+
- Hugging Face Transformers
- Hugging Face Datasets
- Conceptual Captions dataset (practice phase)
