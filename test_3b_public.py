import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use models that don't require special access
models_to_test = [
    "facebook/opt-1.3b",           # 1.3B - we know this works
    "EleutherAI/gpt-neo-1.3B",     # 1.3B - public
    "microsoft/DialoGPT-large",    # 774M - fallback
    "gpt2-large"                   # 774M - always accessible
]

for model_name in models_to_test:
    print(f"Testing: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.cuda()
        params = sum(p.numel() for p in model.parameters())
        memory = torch.cuda.memory_allocated() / 1024**3
        print(f"  ✅ SUCCESS: {params:,} params, {memory:.1f}GB memory")
        
        # Test inference
        inputs = tokenizer("Describe this image:", return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        print(f"  🧪 Inference test passed")
        
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
