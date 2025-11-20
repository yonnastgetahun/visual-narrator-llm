import torch
from PIL import Image

print("🧪 Minimal image test:")
print(f"PyTorch: {torch.__version__}")

try:
    # Test basic image functionality
    img = Image.new('RGB', (224, 224), color='red')
    print(f"✅ PIL Image created: {img.size}")
    
    # Test if we can load a simple model directly
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("✅ Direct model imports: WORKING")
    
    print("🎯 Ready for Phase 7 development!")
    
except Exception as e:
    print(f"❌ Error: {e}")
