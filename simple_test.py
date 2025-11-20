import torch
import transformers

print("🧪 Simple compatibility test:")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")

try:
    # Test basic import
    from transformers import pipeline
    print("✅ Basic imports: WORKING")
    
    # Test simple text pipeline first
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this project!")
    print(f"✅ Text pipeline: {result}")
    
    print("🚀 Environment is functional!")
    
except Exception as e:
    print(f"❌ Error: {e}")
