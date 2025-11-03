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
    """Simple model setup"""
    
    model_name = "facebook/opt-2.7b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
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
    logger.info("🚀 Starting Adjective Training - FULL DATASET")
    
    # Load ALL examples now that we know it works
    train_examples = load_adjective_datasets()
    
    logger.info(f"Using {len(train_examples)} examples for full training")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create dataset
    train_dataset = create_training_dataset(tokenizer, train_examples)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Full training arguments
    training_args = TrainingArguments(
        output_dir="./adjective_dominance_full",
        overwrite_output_dir=True,
        num_train_epochs=3,  # Full training
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=25,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=True,
        fp16=False,  # Keep disabled for stability
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
    logger.info("🔥 Beginning FULL training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("✅ FULL Training complete!")
    
    # Test generation - FIXED DEVICE ISSUE
    test_prompt = "Describe this image: beautiful"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"📝 TEST GENERATION: {generated_text}")
    
    # Test a few more examples
    test_prompts = [
        "Describe this image: vibrant",
        "This picture shows massive",
        "Generate a caption for this peaceful scene:"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=80, 
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"📝 '{prompt}' → {generated}")

if __name__ == "__main__":
    main()
