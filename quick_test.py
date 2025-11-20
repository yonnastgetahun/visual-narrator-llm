import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob
import os

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive']
    return sum(1 for adj in adjectives if adj in text.lower())

# Test the latest model
checkpoints = glob.glob("outputs/phase7_optimized/checkpoint-epoch-*")
if checkpoints:
    model_path = sorted(checkpoints)[-1]
    print(f"🧪 Quick test of: {os.path.basename(model_path)}")
    
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to("cuda")
    
    # Test on 3 random images
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:3]
    
    total_adjectives = 0
    for img_path in test_images:
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model.generate(**inputs, max_length=50)
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            adj_count = count_adjectives(caption)
            total_adjectives += adj_count
            
            print(f"📸 {os.path.basename(img_path)}")
            print(f"   '{caption}'")
            print(f"   Adjectives: {adj_count}\n")
            
        except Exception as e:
            print(f"Error: {e}")
    
    avg_adj = total_adjectives / len(test_images) if test_images else 0
    print(f"📊 CURRENT PERFORMANCE: {avg_adj:.2f} adjectives/description")
    print(f"🎯 TARGET FOR PHASE 7.3: ≥3.0 adjectives/description")
    print(f"📈 IMPROVEMENT NEEDED: {max(0, 3.0 - avg_adj):.2f}")
else:
    print("❌ No model found")
