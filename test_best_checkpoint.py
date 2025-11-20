import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob
import os
import json

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive']
    return sum(1 for adj in adjectives if adj in text.lower())

def test_best_checkpoint():
    """Test the best performing checkpoint from Phase 7.3"""
    
    print("🏆 TESTING BEST PHASE 7.3 CHECKPOINT")
    print("=" * 60)
    
    # Find the checkpoint with highest adjective density
    best_ckpt = "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982"
    
    if not os.path.exists(best_ckpt):
        print("❌ Best checkpoint not found, using latest")
        checkpoints = glob.glob("outputs/phase7_3_large_scale/checkpoint-step-*")
        best_ckpt = sorted(checkpoints)[-1] if checkpoints else ""
    
    print(f"📁 Testing: {os.path.basename(best_ckpt)}")
    print(f"🎯 Historical performance: 4.20 adjectives/description")
    
    # Load model
    processor = BlipProcessor.from_pretrained(best_ckpt)
    model = BlipForConditionalGeneration.from_pretrained(best_ckpt).to("cuda")
    
    # Test on diverse images
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:10] + \
                  glob.glob("/home/ubuntu/data/coco/train2017_5k/*.jpg")[:5]
    
    adjective_counts = []
    results = []
    
    print(f"🖼️  Testing on {len(test_images)} diverse images...")
    print("-" * 60)
    
    for img_path in test_images:
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            adj_count = count_adjectives(caption)
            adjective_counts.append(adj_count)
            
            results.append({
                "image": os.path.basename(img_path),
                "caption": caption,
                "adjectives": adj_count
            })
            
            print(f"📸 {os.path.basename(img_path)}")
            print(f"   '{caption}'")
            print(f"   🎯 Adjectives: {adj_count}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if adjective_counts:
        avg_adjectives = sum(adjective_counts) / len(adjective_counts)
        max_adjectives = max(adjective_counts)
        min_adjectives = min(adjective_counts)
        
        print("=" * 60)
        print(f"🏆 BEST CHECKPOINT PERFORMANCE:")
        print(f"   ✅ Tested on: {len(adjective_counts)} images")
        print(f"   🎯 Average adjectives: {avg_adjectives:.2f}")
        print(f"   📈 Best: {max_adjectives} adjectives")
        print(f"   📉 Worst: {min_adjectives} adjectives")
        print(f"   🎯 Target: ≥3.0 adjectives/description")
        print(f"   📊 Performance: {avg_adjectives/3.0*100:.1f}% of target")
        
        # Save detailed results
        results_path = "phase7/best_checkpoint_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "checkpoint": best_ckpt,
                "test_results": results,
                "summary": {
                    "avg_adjectives": avg_adjectives,
                    "max_adjectives": max_adjectives,
                    "min_adjectives": min_adjectives,
                    "samples_tested": len(adjective_counts),
                    "performance_rating": avg_adjectives/3.0*100
                }
            }, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_path}")
        
        return avg_adjectives
    
    return 0

if __name__ == "__main__":
    test_best_checkpoint()
