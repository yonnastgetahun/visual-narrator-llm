#!/usr/bin/env python3
"""
Baseline training script for Visual Narrator LLM
Practice script using COCO dataset
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

def main():
    print("🚀 Starting Visual Narrator LLM - Practice Phase")
    
    # Load a small model for practice
    model_name = "microsoft/DialoGPT-small"  # Good for practice
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # TODO: Add data loading and training logic
    print("📊 Next step: Implement data loading from COCO...")

if __name__ == "__main__":
    main()
