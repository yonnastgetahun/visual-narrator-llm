import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_download():
    """Test downloading and loading Qwen 2.5-3B with proper tokenizer"""
    
    # Try both base and instruct versions
    model_options = [
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct"
    ]
    
    for model_id in model_options:
        try:
            logger.info(f"🚀 Testing {model_id}")
            
            # Download tokenizer with trust_remote_code for Qwen
            logger.info("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            logger.info("✅ Tokenizer loaded successfully")
            
            # Download model
            logger.info("📥 Downloading model (fp16)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✅ Model loaded successfully")
            
            # Test basic inference
            logger.info("🧪 Testing basic inference...")
            prompt = "Describe this image: beautiful"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=80,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"📝 Test generation: {generated}")
            
            # Memory usage
            memory_gb = model.get_memory_footprint() / 1e9
            logger.info(f"💾 Model memory: {memory_gb:.2f} GB")
            
            logger.info(f"🎉 {model_id} setup SUCCESSFUL!")
            return model_id, tokenizer, model
            
        except Exception as e:
            logger.error(f"❌ {model_id} failed: {e}")
            continue
    
    return None, None, None

if __name__ == "__main__":
    model_id, tokenizer, model = test_qwen_download()
    if model_id:
        print(f"🎉 SUCCESS: {model_id} is ready for Phase 6.2!")
    else:
        print("❌ All Qwen 2.5-3B options failed")
