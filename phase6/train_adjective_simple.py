import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import json
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_adjective_datasets():
    """Load our adjective datasets"""
    examples = []
    
    # Load just the basic adjectives for now
    file_path = 'adjective_data/forced_adjectives.jsonl'
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data['text'])
    
    logger.info(f"Loaded {len(examples)} adjective examples")
    return examples

def setup_model_and_tokenizer():
    """Simple model setup without complex LoRA stacking"""
    
    # Use a fresh base model to avoid conflicts
    model_name = "facebook/opt-2.7b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simple LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Smaller for stability
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_training_dataset(tokenizer, examples):
    """Simple dataset creation"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=256,
        )
    
    dataset = Dataset.from_dict({'text': examples})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main():
    logger.info("🚀 Starting Simple Adjective Training")
    
    # Load just 100 examples for testing
    all_examples = load_adjective_datasets()
    train_examples = all_examples[:100]  # Small batch for testing
    
    logger.info(f"Using {len(train_examples)} examples for initial test")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create dataset
    train_dataset = create_training_dataset(tokenizer, train_examples)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Simple training arguments - NO FP16 for stability
    training_args = TrainingArguments(
        output_dir="./adjective_simple_test",
        overwrite_output_dir=True,
        num_train_epochs=2,  # Short test run
        per_device_train_batch_size=2,  # Small batch
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=1,
        remove_unused_columns=True,
        fp16=False,  # DISABLE FP16 for stability
        dataloader_pin_memory=False,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training!
    logger.info("🔥 Beginning training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("✅ Training complete!")
    
    # Test generation
    test_prompt = "Describe this image: beautiful"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            num_return_sequences=1,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"📝 Test generation: {generated_text}")

if __name__ == "__main__":
    main()
