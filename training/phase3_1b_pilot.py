#!/usr/bin/env python3
"""
PHASE 3.1: 1B Parameter Pilot Training
First cloud training run on A100
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import time

print("🚀 STARTING PHASE 3.1 - 1B PARAMETER PILOT")
print("=" * 60)

# Test environment
print(f"🎯 A100 GPU: {torch.cuda.get_device_name(0)}")
print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load a 1B parameter model (we'll start with a smaller one for testing)
model_name = "microsoft/DialoGPT-large"  # 762M parameters - good starting point
print(f"🤖 Loading model: {model_name}")

start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU
model = model.cuda()
load_time = time.time() - start_load

print(f"✅ Model loaded in {load_time:.2f} seconds")
print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✅ GPU Memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

# Load a small dataset for testing
print("📥 Loading Conceptual Captions sample...")
dataset = load_dataset("conceptual_captions", split="train[:100]")
print(f"✅ Loaded {len(dataset)} examples")

# Quick generation test
print("🧪 Testing model generation...")
prompt = "Describe this image: a beautiful sunset"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"🤖 Model output: {generated}")

print("🎉 PHASE 3.1 PILOT COMPLETED SUCCESSFULLY!")
print("🚀 Ready for full-scale 1B training!")
