license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
tags:
- image-captioning
- visual-narrator
- text-generation
- causal-lm
model-index:
- name: visual-narrator-llm
  results: []
  

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
---
language: en
license: apache-2.0
library_name: transformers
tags:
- image-captioning
- visual-narrator
- text-generation
- causal-lm
---

# Visual Narrator LLM 🎥→📝

A specialized language model fine-tuned for generating vivid image descriptions, trained as part of a project to create a SOTA 3B parameter visual narration model.

## Model Description

This model is part of a research project to create state-of-the-art small language models specialized in visual narration. The model was fine-tuned on Conceptual Captions dataset to generate coherent, creative descriptions of images.

- **Architecture:** GPT2 (causal language model)
- **Parameters:** 124 million
- **Training Data:** 2,000 Conceptual Captions examples
- **Training Loss:** 8.14 → 1.09 (87% reduction)

## Intended Use

- Generating image descriptions for accessibility
- Visual storytelling and narration
- Training foundation for larger visual-language models
- Research in specialized small language models

## Training Results

### Phase 1 (DialoGPT-small):
- Loss: 6.13 → 1.33 (79% reduction)
- Training time: 1 minute 48 seconds

### Phase 2 (GPT2 - this model):
- Loss: 8.14 → 1.09 (87% reduction) 
- Training time: 3 minutes 39 seconds

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Ytgetahun/visual-narrator-llm")
model = AutoModelForCausalLM.from_pretrained("Ytgetahun/visual-narrator-llm")

prompt = "Describe this image: a sunset"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
description = tokenizer.decode(outputs[0], skip_special_tokens=True) 
