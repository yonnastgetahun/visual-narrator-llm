from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🧪 TESTING FIXED PHASE 2 MODEL")
print("=" * 50)

# Test both models for comparison
models_to_test = [
    ("./outputs/first_run", "Phase 1 Model"),
    ("./outputs/phase2_fixed", "Phase 2 Fixed Model"),
    ("./outputs/phase2_run", "Phase 2 Broken Model")
]

test_prompts = [
    "Describe this image: a dog",
    "Describe this image: a city at night",
    "Describe this image: a person riding a",
    "Describe this image: food on a table"
]

for model_path, model_name in models_to_test:
    print(f"\n🔍 Testing: {model_name}")
    print("-" * 30)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=40,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True
                )
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full_text[len(prompt):].strip()
            print(f"   {prompt}")
            print(f"   → {generated}")
            print()
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
