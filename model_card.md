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
