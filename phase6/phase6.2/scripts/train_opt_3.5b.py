import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_adjective_data():
    """Load our proven adjective dataset"""
    examples = []
    with open('data/adjective_data/forced_adjectives.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data['text'])
    
    logger.info(f"📊 Loaded {len(examples)} adjective examples")
    return examples

def setup_opt_3_5b():
    """Setup OPT-3.5B for training"""
    model_id = "facebook/opt-3.5b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Same LoRA config that worked for OPT-2.7B
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def main():
    logger.info("🚀 PHASE 6.2: Starting OPT-3.5B Adjective Training")
    
    # Setup
    model, tokenizer = setup_opt_3_5b()
    train_examples = load_adjective_data()[:200]  # Start with 200 for testing
    
    # Create dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)
    
    dataset = Dataset.from_dict({'text': train_examples})
    train_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir="./opt_3.5b_adjective_test",
        num_train_epochs=1,  # Quick test
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        fp16=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training!
    logger.info("🔥 Beginning OPT-3.5B adjective training...")
    trainer.train()
    
    logger.info("✅ OPT-3.5B training test complete!")
    
    # Test generation after training
    logger.info("🧪 Testing post-training generation...")
    test_prompt = "Describe this image: beautiful"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=80,
            do_sample=True,
            temperature=0.7
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"📝 Post-training generation: {generated}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main()
    print("🎉 PHASE 6.2: OPT-3.5B adjective training SUCCESSFUL!")
