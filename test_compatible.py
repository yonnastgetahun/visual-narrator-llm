import torch
import transformers
import numpy as np
from PIL import Image

print("🎯 COMPATIBLE ENVIRONMENT CHECK:")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"💡 CUDA: {torch.cuda.is_available()}")

try:
    from transformers import pipeline
    print("🚀 Pipelines: WORKING")
    
    # Test image captioning
    captioner = pipeline("image-to-text", 
                        model="nlpconnect/vit-gpt2-image-captioning",
                        device=-1)
    
    test_image = Image.new('RGB', (224, 224), color='purple')
    result = captioner(test_image)
    print(f"📸 Test caption: '{result[0]['generated_text']}'")
    
    print("🎯 PHASE 7: READY TO LAUNCH!")
    
except Exception as e:
    print(f"❌ Error: {e}")
