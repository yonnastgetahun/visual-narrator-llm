from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def evaluate_model(model_path, model_name="Current Model"):
    """Comprehensive model evaluation"""
    print(f"\n🧮 EVALUATING: {model_name}")
    print("=" * 50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Test cases with expected themes
    test_cases = [
        {"prompt": "Describe this image: a dog", "expected": "animal"},
        {"prompt": "Describe this image: a city at night", "expected": "urban"},
        {"prompt": "Describe this image: a beach with", "expected": "coastal"},
        {"prompt": "Describe this image: a person riding a", "expected": "action"},
        {"prompt": "Describe this image: food on a table", "expected": "culinary"}
    ]
    
    print("📝 Generation Test:")
    for i, case in enumerate(test_cases, 1):
        inputs = tokenizer(case["prompt"], return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=60,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=2
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(case["prompt"]):].strip()
        
        print(f"{i}. Prompt: {case['prompt']}")
        print(f"   Generated: {generated}")
        print(f"   Expected theme: {case['expected']}")
        print()
    
    # Test coherence with longer prompts
    print("🔍 Coherence Test:")
    coherence_prompts = [
        "Describe this image: a wedding ceremony with",
        "Describe this image: a sports competition in",
        "Describe this image: a natural disaster showing"
    ]
    
    for prompt in coherence_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.6
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = result[len(prompt):].strip()
        print(f"   '{prompt}' → '{continuation}'")

# Evaluate current model
evaluate_model("./outputs/first_run", "Phase 1 Model (124M)")

print("\n🎯 NEXT: Run 'python training/scale_up_phase.py' to train a larger model!")
print("💡 This will use 5K examples and a 355M parameter model for better quality!")
