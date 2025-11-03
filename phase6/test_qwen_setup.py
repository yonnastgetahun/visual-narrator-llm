import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_download():
    """Test downloading and loading Qwen 2.5-3B"""
    
    # Qwen 2.5-3B model ID
    model_id = "Qwen/Qwen2.5-3B"  # or "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        logger.info(f"🚀 Testing download of {model_id}")
        
        # Test tokenizer download
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Test model download (fp16 to save memory)
        logger.info("📥 Downloading model (fp16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("✅ Model loaded successfully")
        
        # Test basic inference
        logger.info("🧪 Testing basic inference...")
        prompt = "Describe this image: beautiful"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                temperature=0.7
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"📝 Test generation: {generated}")
        
        # Memory usage
        logger.info(f"💾 Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen_download()
    if success:
        print("🎉 Qwen 2.5-3B setup test PASSED!")
    else:
        print("❌ Qwen 2.5-3B setup test FAILED")
