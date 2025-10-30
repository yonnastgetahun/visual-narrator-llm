#!/usr/bin/env python3
"""
Phase 2: Scaling Up Visual Narrator Training
- Larger dataset (10K examples)
- Larger model (GPT2-medium)
- Better evaluation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import evaluate

def prepare_larger_dataset(tokenizer, num_examples=10000):
    """Load a larger dataset for more substantial training"""
    print(f"📥 Loading {num_examples} Conceptual Captions examples...")
    
    dataset = load_dataset("conceptual_captions", split=f"train[:{num_examples}]")
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Format for training
    texts = [f"Describe this image: {example['caption']}" for example in dataset]
    
    # Tokenize
    print("🔤 Tokenizing larger dataset...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    return encodings

def main():
    print("🚀 STARTING PHASE 2: SCALING UP VISUAL NARRATOR")
    print("=" * 60)
    
    # Try a larger model for better performance
    model_name = "gpt2-medium"  # 355M parameters vs 124M
    print(f"🤖 Loading larger model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        print("🔄 Falling back to gpt2")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare larger dataset
    encodings = prepare_larger_dataset(tokenizer, num_examples=5000)  # Start with 5K
    
    # Create dataset
    class CaptionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
            
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    dataset = CaptionDataset(encodings)
    print(f"📊 Dataset prepared: {len(dataset)} samples")
    
    # Training arguments for larger run
    training_args = TrainingArguments(
        output_dir="./outputs/phase2_run",
        overwrite_output_dir=True,
        num_train_epochs=2,  # More epochs for larger dataset
        per_device_train_batch_size=2,  # Smaller batch for larger model
        save_steps=500,
        save_total_limit=3,
        logging_dir='./logs/phase2',
        logging_steps=50,
        report_to=None,
        no_cuda=True,  # CPU for stability
        dataloader_pin_memory=False,
        learning_rate=5e-5,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("🎯 Starting Phase 2 training...")
    print("💡 Training larger model on more data")
    print(f"💻 This will take longer than first run (~5-10 minutes)")
    
    # Start training!
    trainer.train()
    
    print("✅ Phase 2 training completed!")
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained("./outputs/phase2_run")
    print("💾 Phase 2 model saved to ./outputs/phase2_run")

if __name__ == "__main__":
    main()
