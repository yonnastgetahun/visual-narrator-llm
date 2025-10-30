#!/usr/bin/env python3
"""
FIXED Phase 2: Scaling Up Visual Narrator Training
- Proper data preparation for causal LM
- Fixed loss computation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

def prepare_dataset_properly(tokenizer, num_examples=2000):  # Start smaller for testing
    """Proper dataset preparation for causal LM"""
    print(f"📥 Loading {num_examples} Conceptual Captions examples...")
    
    dataset = load_dataset("conceptual_captions", split=f"train[:{num_examples}]")
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Format properly for causal LM training
    texts = []
    for example in dataset:
        # Create proper training examples
        text = f"Describe this image: {example['caption']}"
        texts.append(text)
    
    # Tokenize properly - crucial step!
    print("🔤 Tokenizing dataset (this may take a minute)...")
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    return tokenized

def main():
    print("🚀 STARTING FIXED PHASE 2 TRAINING")
    print("=" * 60)
    
    # Use a reliable model
    model_name = "gpt2"  # Start with smaller model for testing
    print(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # CRITICAL: Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare dataset
    encodings = prepare_dataset_properly(tokenizer, num_examples=2000)
    
    # Create dataset with proper labels
    class FixedCaptionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            item = {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx].clone()  # CRITICAL: labels same as input_ids
            }
            return item
            
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    dataset = FixedCaptionDataset(encodings)
    print(f"📊 Dataset prepared: {len(dataset)} samples")
    
    # Test forward pass with loss
    print("🧪 Testing forward pass with loss computation...")
    sample = dataset[0]
    # Convert to proper format for model
    sample_batch = {
        'input_ids': sample['input_ids'].unsqueeze(0),
        'attention_mask': sample['attention_mask'].unsqueeze(0),
        'labels': sample['labels'].unsqueeze(0)
    }
    
    with torch.no_grad():
        outputs = model(**sample_batch)
    
    print(f"✅ Loss test: {outputs.loss.item():.4f} (should NOT be 0.0)")
    
    # Training arguments - optimized
    training_args = TrainingArguments(
        output_dir="./outputs/phase2_fixed",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Just 1 epoch for testing
        per_device_train_batch_size=4,
        save_steps=500,
        logging_dir='./logs/phase2_fixed',
        logging_steps=50,
        report_to=None,
        use_cpu=True,  # Use this instead of no_cuda
        dataloader_pin_memory=False,
        learning_rate=5e-5,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("🎯 Starting FIXED training...")
    print("💡 Monitoring for proper loss decrease...")
    
    # Start training
    trainer.train()
    
    print("✅ Fixed training completed!")
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained("./outputs/phase2_fixed")
    print("💾 Model saved to ./outputs/phase2_fixed")

if __name__ == "__main__":
    main()
