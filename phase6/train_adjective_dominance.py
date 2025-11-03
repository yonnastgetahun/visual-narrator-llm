import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import json
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_adjective_datasets():
    """Load our three-tier adjective datasets"""
    datasets = {}
    
    for dataset_name in ['forced_adjectives', 'multi_adjective_stacking', 'complex_scenes']:
        file_path = f'adjective_data/{dataset_name}.jsonl'
        examples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append(data['text'])
        
        datasets[dataset_name] = examples
        logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
    
    return datasets

def setup_model_and_tokenizer():
    """Setup model with LoRA for efficient training"""
    
    # Use your latest production model - UPDATE THIS PATH
    model_path = "../phase4/production/lora_model"  # ⚠️ CHOOSE ONE FROM ABOVE
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if it's a LoRA model or base model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # It's a LoRA model - load base model then apply LoRA
        base_model_path = "facebook/opt-2.7b"  # Adjust if different base
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # It's a full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # New LoRA configuration for adjective training
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.05,
        bias="lora_only",
        task_type="CAUSAL_LM"
    )
    
    # Apply new LoRA for adjective specialization
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_training_dataset(tokenizer, examples, max_length=512):
    """Create training dataset from examples"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({'text': examples})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main():
    logger.info("🚀 Starting Adjective Dominance Training")
    
    # Load datasets
    datasets = load_adjective_datasets()
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Start with BASIC adjectives only (Progressive Strategy)
    logger.info("🎯 Phase 1: Training on Basic Adjectives")
    basic_dataset = create_training_dataset(tokenizer, datasets['forced_adjectives'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./adjective_dominance_phase1",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=True,
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=basic_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training!
    logger.info("🔥 Beginning training...")
    trainer.train()
    
    # Save phase 1 model
    trainer.save_model()
    logger.info("✅ Phase 1 training complete!")
    
    # Quick test
    test_prompt = "Describe this image: beautiful"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"📝 Test generation: {generated_text}")

if __name__ == "__main__":
    main()
