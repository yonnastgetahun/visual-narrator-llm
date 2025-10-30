#!/usr/bin/env python3
"""
First Real Training Script - Visual Narrator LLM
Using Conceptual Captions dataset for practice
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

def prepare_conceptual_data():
    """Load and prepare conceptual captions data"""
    print("📥 Loading Conceptual Captions dataset...")
    
    # Load a small subset for practice
    dataset = load_dataset("conceptual_captions", split="train[:1000]")
    print(f"✅ Loaded {len(dataset)} examples")
    
    # For now, we'll just use the captions (not the images)
    # Format: "Describe this image: [caption]"
    texts = [f"Describe this image: {example['caption']}" for example in dataset]
    
    return texts

def main():
    print("🚀 Starting First Real Training Run - Visual Narrator LLM")
    print("=" * 60)
    
    # Load a small, efficient model for practice
    model_name = "microsoft/DialoGPT-small"
    print(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare data
    texts = prepare_conceptual_data()
    
    # Tokenize the data
    print("🔤 Tokenizing dataset...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    
    # Create dataset
    class CaptionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    dataset = CaptionDataset(encodings)
    print(f"📊 Dataset prepared: {len(dataset)} samples")
    
    # Test a forward pass
    print("🧪 Testing model forward pass...")
    sample = dataset[0]
    with torch.no_grad():
        outputs = model(input_ids=sample['input_ids'].unsqueeze(0), 
                       attention_mask=sample['attention_mask'].unsqueeze(0))
    
    print(f"✅ Forward pass successful! Loss: {outputs.loss.item() if outputs.loss else 'N/A'}")
    
    # Set up training arguments (minimal for practice)
    training_args = TrainingArguments(
        output_dir="./outputs/first_run",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Just one epoch for testing
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        report_to=None,  # Disable external logging for now
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("🎯 Starting training...")
    print("💡 This is a PRACTICE run to verify the pipeline works")
    print("💡 We'll train for just 1 epoch on 1000 examples")
    
    # Start training!
    trainer.train()
    
    print("✅ Training completed successfully!")
    print("🎉 Your Visual Narrator LLM pipeline is WORKING!")
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained("./outputs/first_run")
    print("💾 Model saved to ./outputs/first_run")

if __name__ == "__main__":
    main()
