import torch
import transformers
import huggingface_hub
from PIL import Image

print("🎯 FINAL COMPATIBILITY CHECK:")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ HuggingFace Hub: {huggingface_hub.__version__}")
print(f"💡 CUDA: {torch.cuda.is_available()}")

try:
    from transformers import pipeline
    print("🚀 Pipelines: WORKING")
    
    # Test image captioning
    captioner = pipeline("image-to-text", 
                        model="nlpconnect/vit-gpt2-image-captioning",
                        device=-1)
    
    test_image = Image.new('RGB', (224, 224), color='blue')
    result = captioner(test_image)
    print(f"📸 Test caption: '{result[0]['generated_text']}'")
    
    print("🎯 PHASE 7: ENVIRONMENT READY!")
    print("🚀 Launching Phase 7 development...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Need to fix version compatibility")
