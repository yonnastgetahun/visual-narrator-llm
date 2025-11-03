import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel
import json
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_multi_adjective_data():
    """Load multi-adjective examples"""
    examples = []
    file_path = 'adjective_data/multi_adjective_stacking.jsonl'
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data['text'])
    
    logger.info(f"Loaded {len(examples)} multi-adjective examples")
    return examples

def main():
    logger.info("🚀 Phase 2: Multi-Adjective Training")
    
    # Load your trained model
    model_path = "./adjective_dominance_full"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load multi-adjective data
    train_examples = load_multi_adjective_data()
    
    # Create dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)
    
    dataset = Dataset.from_dict({'text': train_examples})
    train_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./adjective_multi_enhanced",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("🔥 Training multi-adjective patterns...")
    trainer.train()
    trainer.save_model()
    logger.info("✅ Multi-adjective training complete!")

if __name__ == "__main__":
    main()
