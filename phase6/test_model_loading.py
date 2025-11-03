from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Test loading
base_model = "facebook/opt-2.7b"
adapter_path = "./adjective_complete_model"

print("Testing model loading...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_path)

print("✅ Model loaded successfully!")
print(f"Model device: {model.device}")

# Quick generation test
prompt = "Describe this beautiful vibrant"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_length=60, do_sample=True)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Test generation: {generated}")
