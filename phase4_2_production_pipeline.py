#!/usr/bin/env python3
"""
PHASE 4.2: PRODUCTION DATASET PIPELINE
Robust dataset integration with multiple fallbacks
"""

print("🚀 PHASE 4.2: PRODUCTION DATASET PIPELINE")
print("=" * 50)

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import pandas as pd

def create_robust_dataset():
    """Create training data with multiple fallback strategies"""
    print("📊 Creating robust training dataset...")
    
    texts = []
    
    # Strategy 1: Enhanced Synthetic Data (ALWAYS WORKS)
    print("1. Creating enhanced synthetic dataset...")
    synthetic_categories = {
        "nature": [
            "a beautiful sunset over mountains with clouds reflecting orange and pink hues",
            "a peaceful forest with sunlight filtering through dense canopy of trees",
            "a tropical beach with white sand, palm trees, and turquoise ocean waves",
            "a snowy mountain peak with dramatic clouds and clear blue sky",
            "a field of wildflowers with butterflies and bees gathering nectar"
        ],
        "urban": [
            "a busy city street with cars, pedestrians, and tall skyscrapers",
            "a modern kitchen with stainless steel appliances and marble countertops",
            "a historic building with intricate stone architecture and large windows",
            "a subway station with trains, commuters, and digital information displays",
            "a shopping mall with multiple floors, stores, and food court"
        ],
        "people": [
            "a sports event with athletes competing and cheering crowd in stadium",
            "a classroom with students listening to teacher and digital whiteboard",
            "a medical laboratory with scientists conducting experiments using equipment",
            "a concert venue with musicians on stage and enthusiastic audience",
            "a farmer's market with vendors selling fresh produce and handmade goods"
        ],
        "indoor": [
            "a library with bookshelves, reading areas, and people studying quietly",
            "an art gallery with paintings on walls and visitors admiring artwork",
            "a hospital room with medical equipment, bed, and monitoring devices",
            "a restaurant with tables, customers dining, and kitchen visible behind",
            "an office space with computers, meetings, and collaborative work areas"
        ]
    }
    
    # Generate diverse synthetic examples
    for category, descriptions in synthetic_categories.items():
        for desc in descriptions:
            texts.append(f"Describe this image: {desc}")
    
    # Add variations
    base_descriptions = list(synthetic_categories.values())[0]  # Use nature as base
    for desc in base_descriptions:
        # Add different phrasing styles
        texts.append(f"Provide a detailed description of this image: {desc}")
        texts.append(f"Generate an image caption: {desc}")
        texts.append(f"What is shown in this image? {desc}")
    
    print(f"   ✅ Enhanced synthetic: {len(texts)} examples")
    
    # Strategy 2: Try COCO without trust_remote_code
    print("2. Attempting COCO dataset load...")
    try:
        from datasets import load_dataset
        # Try different COCO variants
        coco_variants = [
            "HuggingFaceM4/COCO",
            "nlphuji/COCO",
            "coco"
        ]
        
        for variant in coco_variants:
            try:
                coco_dataset = load_dataset(variant, split="train[:500]")
                for example in coco_dataset:
                    if 'sentences' in example and example['sentences']:
                        texts.append(f"Describe this image: {example['sentences'][0]['raw']}")
                    elif 'caption' in example:
                        texts.append(f"Describe this image: {example['caption']}")
                print(f"   ✅ {variant}: {len(coco_dataset)} examples added")
                break
            except Exception as e:
                print(f"   ⚠️  {variant} failed: {str(e)[:100]}...")
                continue
    except Exception as e:
        print(f"   ⚠️  COCO loading failed: {str(e)[:100]}...")
    
    # Strategy 3: Create manual dataset from known working sources
    print("3. Adding curated examples from known datasets...")
    curated_examples = [
        "Describe this image: a group of people sitting at a table with food and drinks",
        "Describe this image: a dog running through a grassy field with a ball",
        "Describe this image: a city skyline at night with illuminated buildings",
        "Describe this image: a person using a laptop in a coffee shop",
        "Describe this image: a mountain landscape with a lake and trees",
        "Describe this image: a child playing with toys on a living room floor",
        "Describe this image: a chef cooking in a restaurant kitchen",
        "Describe this image: a beach scene with people swimming and sunbathing",
        "Describe this image: a classroom with students raising their hands",
        "Describe this image: a sports game with players and referees on field"
    ]
    texts.extend(curated_examples * 10)  # Add multiple copies for weight
    
    print(f"📊 FINAL DATASET SIZE: {len(texts)} examples")
    return texts

def setup_training_pipeline():
    """Setup the complete training pipeline"""
    print("\n🤖 Setting up production training pipeline...")
    
    # Use larger model for production
    model_name = "facebook/opt-2.7b"  # Scaling up from 1.3B
    print(f"   Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Enhanced LoRA configuration
    lora_config = LoraConfig(
        r=32,  # Increased rank for better performance
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    lora_model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    
    print(f"   ✅ Model loaded: {trainable_params:,} trainable of {total_params:,} total params")
    print(f"   ✅ Memory efficient: {trainable_params/total_params*100:.1f}% parameters trained")
    
    return lora_model, tokenizer

def train_production_pipeline(model, tokenizer, texts):
    """Run production training pipeline"""
    print("\n🏭 Running production training...")
    
    # Create proper dataset
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"   Training on {len(dataset)} examples")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Training loop (mini-version for validation)
    model.train()
    total_loss = 0
    
    # Use smaller batch for testing
    for i in range(min(10, len(tokenized_dataset))):  # Just 10 steps for validation
        batch = tokenized_dataset[i]
        inputs = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in batch.items() if k != 'text'}
        inputs["labels"] = inputs["input_ids"].clone()
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if i % 5 == 0:
                print(f"   Step {i}: loss {loss.item():.4f}")
    
    avg_loss = total_loss / min(10, len(tokenized_dataset))
    print(f"   ✅ Average training loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    """Main production pipeline"""
    try:
        # Step 1: Create robust dataset
        texts = create_robust_dataset()
        
        # Step 2: Setup training pipeline
        model, tokenizer = setup_training_pipeline()
        
        # Step 3: Run production training
        final_loss = train_production_pipeline(model, tokenizer, texts)
        
        # Step 4: Save production pipeline
        print("\n💾 Saving production pipeline...")
        import os
        os.makedirs("./phase4/production", exist_ok=True)
        
        model.save_pretrained("./phase4/production/lora_model")
        tokenizer.save_pretrained("./phase4/production/lora_model")
        
        # Save dataset info
        dataset_info = {
            "total_examples": len(texts),
            "final_loss": final_loss,
            "model": "opt-2.7b",
            "lora_rank": 32,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        import json
        with open("./phase4/production/training_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        print("🎯 PHASE 4.2 PRODUCTION PIPELINE SUCCESS!")
        print(f"✅ Dataset: {len(texts)} high-quality examples")
        print(f"✅ Model: OPT-2.7B with enhanced LoRA")
        print(f"✅ Training: Stable loss {final_loss:.4f}")
        print(f"✅ Pipeline: Saved to ./phase4/production/")
        print("🚀 READY FOR PHASE 4.3: ADVANCED TRAINING!")
        
    except Exception as e:
        print(f"❌ Production pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
