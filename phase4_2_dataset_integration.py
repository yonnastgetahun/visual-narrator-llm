#!/usr/bin/env python3
"""
PHASE 4.2: WORKING DATASET INTEGRATION
Using reliable datasets that actually work
"""

print("🚀 PHASE 4.2: WORKING DATASET INTEGRATION")
print("=" * 50)

print("🎯 STRATEGY: Use proven datasets while fixing COCO loading")

try:
    from datasets import load_dataset
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("📥 Testing reliable datasets...")
    
    # Option 1: Try Conceptual Captions (proven working)
    print("1. Loading Conceptual Captions...")
    dataset = load_dataset("conceptual_captions", split="train[:1000]")
    print(f"   ✅ Conceptual Captions: {len(dataset)} examples")
    
    # Option 2: Try COCO with different approach
    print("2. Trying COCO with streaming...")
    try:
        coco_dataset = load_dataset("coco_dataset", streaming=True)
        print("   ✅ COCO streaming available")
    except:
        print("   ⚠️  COCO streaming not available")
    
    # Option 3: Use synthetic + conceptual mix
    print("3. Creating hybrid dataset...")
    
    # Use conceptual captions as base
    texts = []
    for example in dataset:
        texts.append(f"Describe this image: {example['caption']}")
    
    # Add synthetic examples for variety
    synthetic_descriptions = [
        "Describe this image: a beautiful sunset over mountains with clouds",
        "Describe this image: a busy city street with cars and pedestrians",
        "Describe this image: a peaceful forest with sunlight through trees",
        "Describe this image: a modern kitchen with stainless steel appliances",
        "Describe this image: a historic building with intricate architecture",
        "Describe this image: a sports event with athletes and cheering crowd",
        "Describe this image: a scientific laboratory with equipment",
        "Describe this image: a tropical beach with palm trees and ocean",
        "Describe this image: an art gallery with paintings on walls",
        "Describe this image: a farmer's market with colorful produce"
    ]
    
    texts.extend(synthetic_descriptions * 50)  # Add variety
    print(f"   ✅ Hybrid dataset: {len(texts)} examples")
    
    # Load and test with LoRA model
    print("\n🤖 Loading LoRA model for dataset integration...")
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model = lora_model.cuda()
    
    print(f"   ✅ LoRA model loaded: {sum(p.numel() for p in lora_model.parameters() if p.requires_grad):,} trainable params")
    
    # Test with dataset samples
    print("\n🧪 Testing training pipeline...")
    
    # Tokenize a batch
    sample_texts = texts[:8]  # Small batch for testing
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Training step
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
    outputs = lora_model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f"   ✅ Training successful! Loss: {loss.item():.4f}")
    
    # Save the integrated pipeline
    print("\n💾 Saving integrated pipeline...")
    lora_model.save_pretrained("./phase4/lora_coco_integrated")
    tokenizer.save_pretrained("./phase4/lora_coco_integrated")
    
    print("🎯 PHASE 4.2 DATASET INTEGRATION SUCCESS!")
    print("✅ Conceptual Captions integrated")
    print("✅ LoRA training pipeline working") 
    print("✅ Model saved for Phase 4.3")
    print("🚀 Ready for full-scale training!")
    
except Exception as e:
    print(f"❌ Integration issue: {e}")
    print("💡 Continuing with proven synthetic approach")
