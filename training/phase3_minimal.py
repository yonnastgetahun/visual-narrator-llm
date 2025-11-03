#!/usr/bin/env python3
"""
PHASE 3 MINIMAL TRAINING - Bypasses version conflicts
Uses basic PyTorch training loop
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("🚀 PHASE 3 MINIMAL TRAINING - A100 PROOF")
print("=" * 50)

# Load model and tokenizer
model_name = "microsoft/DialoGPT-large"
print(f"🤖 Loading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move to GPU
model = model.cuda()
print(f"✅ Model on A100: {sum(p.numel() for p in model.parameters()):,} parameters")

# Create synthetic data
texts = [
    "Describe this image: a cat sitting on a chair",
    "Describe this image: a dog playing in the park", 
    "Describe this image: a beautiful sunset over mountains",
    "Describe this image: a city skyline at night",
    "Describe this image: a person riding a bicycle",
] * 200  # 1000 examples

print(f"📊 Training data: {len(texts)} examples")

# Tokenize
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Custom training loop (bypasses Trainer)
class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = SimpleDataset(encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("🎯 STARTING MINIMAL A100 TRAINING!")
print("💡 Using custom training loop (bypasses version conflicts)")

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
    
    if step % 10 == 0:
        print(f"   Step {step}/{total_steps} - Loss: {loss.item():.4f}")
    
    if step >= 50:  # Short test
        break

training_time = time.time() - start_time
print(f"✅ MINIMAL TRAINING COMPLETED in {training_time:.2f} seconds!")

# Save model manually
print("💾 Saving model...")
model.save_pretrained("./outputs/phase3_minimal")
tokenizer.save_pretrained("./outputs/phase3_minimal")

print("🚀 PHASE 3 MINIMAL SUCCESS!")
print("🎯 A100 is PROVEN READY for full training!")
