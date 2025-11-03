import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("🚀 Final Qwen 2.5-3B Test")
        
        # Test the base model
        model_id = "Qwen/Qwen2.5-3B"
        
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer loaded")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Padding token set")
        
        logger.info("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Model loaded")
        
        # Quick generation test
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
        
        # Memory info
        memory_gb = model.get_memory_footprint() / 1e9
        logger.info(f"💾 Memory usage: {memory_gb:.2f} GB")
        
        logger.info("🎉 QWEN 2.5-3B SETUP SUCCESSFUL!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 PHASE 6.2: Qwen 2.5-3B READY FOR TRAINING!")
    else:
        print("❌ Qwen setup failed - time to pivot")
