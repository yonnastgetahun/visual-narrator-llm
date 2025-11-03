import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_opt_3_5b():
    """Test OPT-3.5B - proven compatible architecture"""
    
    model_id = "facebook/opt-3.5b"
    
    try:
        logger.info(f"🚀 Testing {model_id}")
        
        # Load tokenizer (no special dependencies needed)
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Padding token set")
        
        # Load model
        logger.info("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("✅ Model loaded successfully")
        
        # Test generation
        logger.info("🧪 Testing generation...")
        prompt = "Describe this image: beautiful vibrant sunset"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"📝 Generated: {generated}")
        
        # Memory usage
        memory_gb = model.get_memory_footprint() / 1e9
        logger.info(f"💾 Memory usage: {memory_gb:.2f} GB")
        
        logger.info("🎉 OPT-3.5B SETUP SUCCESSFUL!")
        return True
        
    except Exception as e:
        logger.error(f"❌ OPT-3.5B failed: {e}")
        return False

if __name__ == "__main__":
    success = test_opt_3_5b()
    if success:
        print("🎉 PHASE 6.2: OPT-3.5B READY FOR TRAINING!")
    else:
        print("❌ OPT-3.5B failed - checking alternatives")
