import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os

def save_model_correctly():
    """Save a model with all necessary files"""
    
    # Setup model (same as training)
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
    
    # Save properly
    save_path = "./adjective_model_proper"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Model saved to {save_path}")
    print("📁 Files created:")
    for file in os.listdir(save_path):
        print(f"   - {file}")

if __name__ == "__main__":
    save_model_correctly()
