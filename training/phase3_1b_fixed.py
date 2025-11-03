#!/usr/bin/env python3
"""
PHASE 3.1: Fixed Training with Reliable Dataset
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import time

print("🚀 PHASE 3.1 - FIXED TRAINING WITH RELIABLE DATASET")
print("=" * 60)

# Use a reliable model
model_name = "microsoft/DialoGPT-large"
print(f"🤖 Loading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Critical: Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move to GPU
model = model.cuda()
print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Use a RELIABLE dataset that always works
print("📥 Loading reliable dataset (COCO captions)...")
try:
    # Try COCO dataset - very reliable
    dataset = load_dataset("ydshieh/coco_dataset_script", "2017", split="train[:1000]")
    texts = [f"Describe this image: {example['caption']}" for example in dataset]
    print(f"✅ Loaded COCO: {len(texts)} examples")
except:
    # Fallback: Create synthetic data
    print("🔄 Using synthetic training data...")
    texts = [
        "Describe this image: a cat sitting on a chair",
        "Describe this image: a dog playing in the park", 
        "Describe this image: a beautiful sunset over mountains",
        "Describe this image: a city skyline at night",
        "Describe this image: a person riding a bicycle",
        "Describe this image: food on a table",
        "Describe this image: a car driving on a road",
        "Describe this image: a beach with waves",
        "Describe this image: a forest with tall trees",
        "Describe this image: a bird flying in the sky"
    ] * 100  # Repeat to get enough examples
    print(f"✅ Created synthetic: {len(texts)} examples")

# Tokenize
print("🔤 Tokenizing dataset...")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Create dataset with labels
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TrainingDataset(encodings)
print(f"📊 Training dataset: {len(train_dataset)} samples")

# Training arguments optimized for A100
training_args = TrainingArguments(
    output_dir="./outputs/phase3_1b",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_dir='./logs/phase3_1b',
    logging_steps=50,
    report_to=None,
    learning_rate=5e-5,
    fp16=True,  # Mixed precision on A100
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("🎯 STARTING A100 CLOUD TRAINING!")
print("💡 First real training on cloud GPU!")
print("⏰ Estimated time: 2-5 minutes")

# Start training!
trainer.train()

print("🎉 PHASE 3.1 TRAINING COMPLETED!")
print("💾 Saving model...")
trainer.save_model()
tokenizer.save_pretrained("./outputs/phase3_1b")

print("🚀 PHASE 3.1 SUCCESS! Ready for scaling!")
