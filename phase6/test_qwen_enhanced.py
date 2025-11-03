import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def test_qwen_enhanced():
    """Enhanced Qwen test with detailed debugging"""
    
    model_options = [
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B",  # Fallback to smaller if 3B fails
        "Qwen/Qwen2.5-0.5B"   # Smallest for testing
    ]
    
    for model_id in model_options:
        try:
            logger.info(f"🚀 Testing {model_id}")
            
            # Test with different tokenizer approaches
            logger.info("📥 Attempting tokenizer download...")
            
            # Approach 1: Standard with trust_remote_code
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True  # Try fast tokenizer
            )
            logger.info("✅ Tokenizer loaded successfully")
            
            # Add padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("✅ Padding token set")
            
            # Load model with various options
            logger.info("📥 Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Model loaded successfully")
            
            # Test inference
            logger.info("🧪 Testing inference...")
            prompt = "Describe this image: beautiful sunset"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"📝 Generation: {generated}")
            
            # Memory and model info
            memory_gb = model.get_memory_footprint() / 1e9
            logger.info(f"💾 Memory: {memory_gb:.2f} GB")
            logger.info(f"🔧 Device: {model.device}")
            
            logger.info(f"🎉 {model_id} COMPLETE SUCCESS!")
            return model_id, tokenizer, model
            
        except Exception as e:
            logger.error(f"❌ {model_id} failed: {str(e)}")
            logger.error(f"🔧 Error type: {type(e).__name__}")
            continue
    
    logger.error("❌ All Qwen models failed")
    return None, None, None

if __name__ == "__main__":
    print("🔧 Enhanced Qwen 2.5 Setup Test")
    model_id, tokenizer, model = test_qwen_enhanced()
    if model_id:
        print(f"🎉 PHASE 6.2 READY: {model_id}")
    else:
        print("❌ Qwen setup failed - consider alternative models")
