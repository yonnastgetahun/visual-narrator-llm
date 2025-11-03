import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import json
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_adjective_data():
    """Load both basic and multi-adjective data"""
    examples = []
    
    # Load basic adjectives
    with open('adjective_data/forced_adjectives.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data['text'])
    
    # Load multi-adjectives
    with open('adjective_data/multi_adjective_stacking.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data['text'])
    
    logger.info(f"Loaded {len(examples)} total adjective examples")
    return examples

def setup_model():
    """Setup fresh model with LoRA"""
    model_name = "facebook/opt-2.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

def main():
    logger.info("🚀 COMPLETE Adjective Training: Basic + Multi-Adjective")
    
    # Load ALL adjective data
    train_examples = load_all_adjective_data()
    logger.info(f"Training on {len(train_examples)} examples")
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Create dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)
    
    dataset = Dataset.from_dict({'text': train_examples})
    train_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./adjective_complete_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=25,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=True,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("🔥 Training on ALL adjective data...")
    trainer.train()
    
    # PROPERLY save the model
    trainer.save_model()
    tokenizer.save_pretrained("./adjective_complete_model")
    logger.info("✅ COMPLETE Adjective training finished!")
    
    # Test generations
    device = model.device
    test_prompts = [
        "Describe this image: beautiful vibrant",
        "This picture shows massive towering",
        "Generate a caption for this peaceful serene scene:",
        "Describe the colorful energetic sports event:"
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
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🎯 '{prompt}' → {generated}")

if __name__ == "__main__":
    main()
