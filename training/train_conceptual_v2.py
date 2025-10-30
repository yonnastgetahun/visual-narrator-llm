#!/usr/bin/env python3
"""
Fixed Training Script - Visual Narrator LLM
Proper loss handling for causal LM training
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

def prepare_conceptual_data(tokenizer):
    """Load and prepare conceptual captions data with proper formatting"""
    print("📥 Loading Conceptual Captions dataset...")
    
    # Load a small subset for practice
    dataset = load_dataset("conceptual_captions", split="train[:1000]")
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Format for causal LM: "Describe this image: [caption]"
    # For causal LM, the input is also the target (we predict next tokens)
    texts = [f"Describe this image: {example['caption']}" for example in dataset]
    
    # Tokenize - for causal LM, we don't need separate labels
    # The model will automatically use input_ids as labels when we set the right arguments
    print("🔤 Tokenizing dataset...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    return encodings

def main():
    print("🚀 Starting Fixed Training Run - Visual Narrator LLM")
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
    encodings = prepare_conceptual_data(tokenizer)
    
    # Create dataset - for causal LM, labels are the same as input_ids
    class CaptionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # For causal LM, labels are the same as input_ids
            item['labels'] = item['input_ids'].clone()
            return item
            
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    dataset = CaptionDataset(encodings)
    print(f"📊 Dataset prepared: {len(dataset)} samples")
    
    # Test a forward pass with labels
    print("🧪 Testing model forward pass with labels...")
    sample = dataset[0]
    with torch.no_grad():
        outputs = model(input_ids=sample['input_ids'].unsqueeze(0), 
                       attention_mask=sample['attention_mask'].unsqueeze(0),
                       labels=sample['labels'].unsqueeze(0))
    
    print(f"✅ Forward pass successful! Loss: {outputs.loss.item():.4f}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/first_run",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        report_to=None,
        # Important: disable MPS for compatibility
        no_cuda=True,  # Force CPU for stability
        dataloader_pin_memory=False,  # Disable pin memory for M1/M2 Macs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("🎯 Starting training...")
    print("💡 Training on CPU for stability (M1/M2 compatibility)")
    
    # Start training!
    trainer.train()
    
    print("✅ Training completed successfully!")
    print("🎉 Your Visual Narrator LLM is now TRAINED!")
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained("./outputs/first_run")
    print("💾 Model saved to ./outputs/first_run")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_prompt = "Describe this image:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Model output: {generated_text}")

if __name__ == "__main__":
    main()
