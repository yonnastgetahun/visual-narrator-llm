---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
base_model: gpt2
tags:
  - image-captioning
  - visual-narrator
  - text-generation
  - causal-lm
  - gpt2
  - small-llm
  - accessibility
datasets:
  - conceptual-captions
metrics:
  - loss
  - perplexity
model-index:
  - name: visual-narrator-llm
    results:
      - task:
          name: Text Generation
          type: text-generation
        dataset:
          name: Conceptual Captions (subset)
          type: conceptual-captions
          split: train[:1000]
        metrics:
          - name: Final training loss
            type: loss
            value: 1.33
            verified: false
          - name: Initial loss
            type: loss
            value: 6.13
            verified: false
        config:
          model: DialoGPT-small
          parameters: 124e6
          training_time: "1m 48s"
          notes: "Phase 1 foundation run"
      - task:
          name: Text Generation
          type: text-generation
        dataset:
          name: Conceptual Captions (subset)
          type: conceptual-captions
          split: train[:2000]
        metrics:
          - name: Final training loss
            type: loss
            value: 1.09
            verified: false
          - name: Initial loss
            type: loss
            value: 8.14
            verified: false
        config:
          model: GPT-2 (small)
          parameters: 124e6
          training_time: "3m 39s"
          notes: "Phase 2 scaling + bug fix"
---
# Visual Narrator LLM 🎥→📝

## Project Goal
Create a state-of-the-art 3B parameter LLM specialized in generating vivid audio descriptions for visual media.

## Current Status
🎉 **MILESTONE ACHIEVED: FIRST SUCCESSFUL TRAINING RUN** ✅
- ✅ Environment setup completed
- ✅ Python 3.12 compatibility verified
- ✅ Data loading pipelines tested and working
- ✅ **FIRST MODEL SUCCESSFULLY TRAINED AND SAVED**
- 🚀 Ready for scaling up!

## Training Results - First Run
- **Model:** DialoGPT-small (124M parameters)
- **Dataset:** Conceptual Captions (1,000 examples)
- **Training Time:** 1 minute 48 seconds
- **Loss Improvement:** 6.13 → 1.33 (79% reduction!)
- **Result:** Model learned to generate image descriptions

## Recent Achievements
- Solved Python 3.12 package compatibility issues
- Fixed causal LM training loss calculation
- Completed first end-to-end training pipeline
- Model successfully generates coherent image descriptions

## Next Steps
1. Scale up dataset size
2. Experiment with larger models
3. Add proper evaluation metrics
4. Move toward 3B parameter target

## Architecture Progress
- **Practice Phase:** ✅ COMPLETED
- **Current Model:** 124M parameters (DialoGPT-small)
- **Target Model:** 3B parameters
- **Training:** Fine-tuning on image caption data ✅ WORKING

**Trained Model Location:** `./outputs/first_run/`

## Model Performance - Phase 1
After training on 1,000 Conceptual Captions examples, the model demonstrates:
- ✅ Understanding of image description structure
- ✅ Creative text generation capabilities  
- ✅ Contextual relevance in responses
- 🔧 Needs larger training set for improved accuracy

### Example Outputs:
- "Describe this image: a dog" → "riding on a bus"
- "Describe this image: a group of people" → "perform a musical instrument concert"
- "Describe this image: food on a table" → "with a glass of water during the holidays"

## Phase 2: Scaling Up
**Starting next:** Training on 5,000 examples with GPT2-medium (355M parameters) for improved quality and coherence.
