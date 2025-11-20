import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob
import json

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive']
    return sum(1 for adj in adjectives if adj in text.lower())

def test_baseline_performance():
    """Test baseline performance before enhanced training"""
    
    print("🧪 BASELINE PERFORMANCE TEST")
    print("=" * 50)
    
    # Load the latest checkpoint from previous training
    checkpoints = glob.glob("outputs/phase7_blip_synth_fp16/checkpoint-*")
    if not checkpoints:
        print("❌ No previous checkpoints found")
        return
    
    latest_ckpt = sorted(checkpoints)[-1]
    print(f"📁 Testing checkpoint: {latest_ckpt}")
    
    # Load model
    processor = BlipProcessor.from_pretrained(latest_ckpt)
    model = BlipForConditionalGeneration.from_pretrained(latest_ckpt).to("cuda")
    
    # Test on sample images
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:5]
    
    adjective_counts = []
    
    print(f"🖼️  Testing on {len(test_images)} images...")
    print("-" * 50)
    
    for img_path in test_images:
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model.generate(**inputs, max_length=50)
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            adj_count = count_adjectives(caption)
            adjective_counts.append(adj_count)
            
            print(f"📸 {os.path.basename(img_path)}")
            print(f"   📝 {caption}")
            print(f"   🎯 Adjectives: {adj_count}")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue
    
    if adjective_counts:
        avg_adjectives = sum(adjective_counts) / len(adjective_counts)
        print("=" * 50)
        print(f"📊 BASELINE PERFORMANCE SUMMARY:")
        print(f"   ✅ Tested on: {len(adjective_counts)} images")
        print(f"   🎯 Average adjectives: {avg_adjectives:.2f} per description")
        print(f"   🎯 Target for enhanced training: ≥3.0")
        print(f"   📈 Improvement needed: {3.0 - avg_adjectives:.2f}")
    
    return avg_adjectives if adjective_counts else 0

if __name__ == "__main__":
    test_baseline_performance()
