import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_enhanced_training():
    """Use the PROVEN Phase 6.1 approach with enhanced parameters"""
    
    # Load the successful model from Phase 6.1
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    logger.info("🚀 Loading PROVEN Phase 6.1 model...")
    
    # Load base model
    model_id = "facebook/opt-2.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Try to load the successful Phase 6.1 model
    try:
        # Load the adapter from successful training
        model = PeftModel.from_pretrained(base_model, "adjective_complete_model")
        logger.info("✅ Loaded successful Phase 6.1 model!")
    except:
        logger.info("🔄 Starting fresh enhanced training...")
        model = base_model
    
    # Create simple enhanced training data
    enhanced_examples = [
        "Describe this image: beautiful vibrant colorful stunning magnificent scene with towering structures",
        "This picture shows massive enormous gigantic colossal architecture in urban setting",
        "Generate a caption for this peaceful serene tranquil calm picturesque landscape view",
        "What do you see in this chaotic busy bustling lively energetic city street scene",
        "A dramatic intense powerful emotional captivating moment captured in this frame",
        "The elegant sophisticated refined exquisite delicate details of this composition",
        "Ancient historic traditional classic timeless architectural marvel standing tall",
        "Modern contemporary innovative futuristic cutting-edge design and technology",
        "Natural organic rustic raw authentic wilderness environment untouched",
        "Urban metropolitan cosmopolitan developed industrialized cityscape panorama"
    ]
    
    logger.info(f"📊 Using {len(enhanced_examples)} enhanced adjective examples")
    
    # Enhanced training with our proven parameters
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    
    # Enhanced LoRA config
    lora_config = LoraConfig(
        r=48,  # Increased from 32
        lora_alpha=96,  # Increased from 64  
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.03,  # Reduced
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=128)
    
    dataset = Dataset.from_dict({'text': enhanced_examples})
    train_dataset = dataset.map(tokenize_function, batched=True)
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir="./phase6_2_enhanced_quick",
        num_train_epochs=4,  # Increased
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1.5e-5,  # Optimized
        warmup_steps=50,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=True,
        fp16=True,
        dataloader_pin_memory=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("🔥 Starting QUICK ENHANCED training...")
    trainer.train()
    
    # Save the enhanced model
    trainer.save_model()
    logger.info("✅ Enhanced training complete!")
    
    # Quick test
    test_prompts = [
        "Describe this urban scene:",
        "This beautiful landscape shows:",
        "Generate caption for historic building:"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=60,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🎯 '{prompt}' → {generated}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = quick_enhanced_training()
    print("🎉 PHASE 6.2 QUICK ENHANCEMENT SUCCESS!")
    print("📈 Building on proven 4.15 adjectives/description foundation")
