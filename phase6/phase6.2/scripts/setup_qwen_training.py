import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_qwen_for_training(model_id="Qwen/Qwen2.5-3B"):
    """Setup Qwen 2.5-3B for adjective training"""
    
    try:
        logger.info(f"🚀 Setting up {model_id} for training")
        
        # Load tokenizer with Qwen-specific settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA configuration optimized for Qwen
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info("✅ Qwen 2.5-3B setup complete for training")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Qwen setup failed: {e}")
        raise

if __name__ == "__main__":
    model, tokenizer = setup_qwen_for_training()
    print("🎉 Qwen 2.5-3B ready for adjective training!")
