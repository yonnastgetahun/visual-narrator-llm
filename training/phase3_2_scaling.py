#!/usr/bin/env python3
"""
PHASE 3.2: SCALING TO 1B+ PARAMETERS
Using proven custom training approach
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("🚀 PHASE 3.2: SCALING TO 1B+ PARAMETERS")
print("=" * 50)

# Use a larger model - Qwen1.5-1.8B (actual 1.8B parameters)
model_name = "Qwen/Qwen2.5-1.5B"  # 1.5B parameters
print(f"🤖 Loading larger model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Fix tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Move to GPU
    model = model.cuda()
    print(f"✅ 1.5B Model loaded on A100!")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Could not load 1.5B model: {e}")
    print("🔄 Falling back to DialoGPT-large (774M)")
    model_name = "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.cuda()

print(f"🎯 GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB used")

# Create larger dataset
texts = [
    "Describe this image: a cat sitting on a chair looking out the window",
    "Describe this image: a dog playing with a ball in a sunny park", 
    "Describe this image: a beautiful sunset over snow-capped mountains",
    "Describe this image: a bustling city skyline with tall buildings at dusk",
    "Describe this image: a person riding a bicycle along a scenic country road",
    "Describe this image: a delicious meal served on a wooden table with candles",
    "Describe this image: a car driving through a rainy city street at night",
    "Describe this image: a tropical beach with palm trees and turquoise water",
    "Describe this image: a dense forest with sunlight filtering through trees",
    "Describe this image: a bird soaring high above the clouds in blue sky"
] * 200  # 2000 examples

print(f"📊 Training data: {len(texts)} examples")

# Tokenize
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Dataset
class TrainingDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = TrainingDataset(encodings)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Smaller batch for larger model

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("🎯 STARTING PHASE 3.2 SCALED TRAINING!")
print("💡 Training larger model on more data")

model.train()
total_steps = len(dataloader)
start_time = time.time()

for step, batch in enumerate(dataloader):
    # Move to GPU
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch, labels=batch['input_ids'])
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f"   Step {step}/{total_steps} - Loss: {loss.item():.4f}")
        print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    if step >= 100:  # More steps for larger model
        break

training_time = time.time() - start_time
print(f"✅ PHASE 3.2 SCALED TRAINING COMPLETED in {training_time:.2f} seconds!")

# Save model
print("💾 Saving scaled model...")
model.save_pretrained("./outputs/phase3_2_scaled")
tokenizer.save_pretrained("./outputs/phase3_2_scaled")

print("🚀 PHASE 3.2 SCALING SUCCESS!")
print("🎯 Ready for Phase 3.3: 3B parameter target!")
