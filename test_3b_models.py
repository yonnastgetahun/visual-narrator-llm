import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

models_to_test = [
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B", 
    "microsoft/Phi-3.5-3B-mini",
    "facebook/opt-3.5b"
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
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
