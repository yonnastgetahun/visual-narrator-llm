from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🔍 Diagnosing Phase 2 Training Results")

# Check if model saved
try:
    tokenizer = AutoTokenizer.from_pretrained("./outputs/phase2_run")
    model = AutoModelForCausalLM.from_pretrained("./outputs/phase2_run")
    print("✅ Phase 2 model files exist")
    
    # Test generation
    prompt = "Describe this image: a cat"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=30,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🧪 Test generation: {result}")
    
except Exception as e:
    print(f"❌ Phase 2 model issue: {e}")

# Compare with Phase 1 model
print("\n🔍 Testing Phase 1 model for comparison:")
try:
    tokenizer_p1 = AutoTokenizer.from_pretrained("./outputs/first_run")
    model_p1 = AutoModelForCausalLM.from_pretrained("./outputs/first_run")
    
    inputs = tokenizer_p1("Describe this image: a cat", return_tensors="pt")
    with torch.no_grad():
        outputs = model_p1.generate(**inputs, max_length=30, num_return_sequences=1)
    
    result_p1 = tokenizer_p1.decode(outputs[0], skip_special_tokens=True)
    print(f"Phase 1 model: {result_p1}")
    
except Exception as e:
    print(f"Phase 1 model error: {e}")
