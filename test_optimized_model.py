import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob
import os
import json

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive',
                 'stunning', 'breathtaking', 'captivating', 'mesmerizing']
    return sum(1 for adj in adjectives if adj in text.lower())

def test_optimized_model():
    """Test the optimized model performance"""
    
    print("🧪 OPTIMIZED MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    # Find the latest checkpoint
    checkpoints = glob.glob("outputs/phase7_optimized/checkpoint-epoch-*")
    if not checkpoints:
        print("❌ No optimized checkpoints found")
        return
    
    latest_ckpt = sorted(checkpoints)[-1]
    print(f"📁 Testing checkpoint: {latest_ckpt}")
    
    # Load model
    processor = BlipProcessor.from_pretrained(latest_ckpt)
    model = BlipForConditionalGeneration.from_pretrained(latest_ckpt).to("cuda")
    
    # Test on sample images
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:8]
    
    adjective_counts = []
    results = []
    
    print(f"🖼️  Testing on {len(test_images)} images...")
    print("-" * 60)
    
    for img_path in test_images:
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model.generate(**inputs, max_length=50)
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            adj_count = count_adjectives(caption)
            adjective_counts.append(adj_count)
            
            results.append({
                "image": os.path.basename(img_path),
                "caption": caption,
                "adjectives": adj_count
            })
            
            print(f"📸 {os.path.basename(img_path)}")
            print(f"   📝 {caption}")
            print(f"   🎯 Adjectives: {adj_count}")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue
    
    if adjective_counts:
        avg_adjectives = sum(adjective_counts) / len(adjective_counts)
        max_adjectives = max(adjective_counts)
        min_adjectives = min(adjective_counts)
        
        print("=" * 60)
        print(f"📊 OPTIMIZED MODEL PERFORMANCE:")
        print(f"   ✅ Tested on: {len(adjective_counts)} images")
        print(f"   🎯 Average adjectives: {avg_adjectives:.2f} per description")
        print(f"   📈 Best: {max_adjectives} adjectives")
        print(f"   📉 Worst: {min_adjectives} adjectives")
        print(f"   🎯 Target: ≥3.0 adjectives/description")
        print(f"   📊 Progress: {avg_adjectives/3.0*100:.1f}% of target")
        
        # Save results
        results_path = "phase7/optimized_test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "checkpoint": latest_ckpt,
                "test_results": results,
                "summary": {
                    "avg_adjectives": avg_adjectives,
                    "max_adjectives": max_adjectives,
                    "min_adjectives": min_adjectives,
                    "samples_tested": len(adjective_counts)
                }
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
    
    return avg_adjectives if adjective_counts else 0

if __name__ == "__main__":
    test_optimized_model()
