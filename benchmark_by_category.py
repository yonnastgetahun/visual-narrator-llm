import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import pandas as pd

def count_adjectives(text):
    adjectives = [
        'vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden',
        'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
        'majestic', 'luminous', 'textured', 'atmospheric', 'expressive'
    ]
    return sum(1 for adj in adjectives if adj in text.lower())

def benchmark_by_category():
    """Benchmark across different image categories"""
    
    print("🎯 CATEGORY-SPECIFIC BENCHMARKING")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load our best model
    our_model_path = "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982"
    our_processor = BlipProcessor.from_pretrained(our_model_path)
    our_model = BlipForConditionalGeneration.from_pretrained(our_model_path).to(device)
    
    # Define test categories with sample images
    categories = {
        "Landscapes": ["coco_downloaded_00000000.jpg", "coco_downloaded_00000017.jpg"],
        "Portraits": ["coco_downloaded_00000001.jpg", "coco_downloaded_00000002.jpg"],
        "Urban Scenes": ["coco_downloaded_00000003.jpg", "coco_downloaded_00000010.jpg"],
        "Objects": ["img_001.jpg", "img_015.jpg", "img_020.jpg"],
        "Indoor Scenes": ["img_024.jpg", "img_028.jpg"]
    }
    
    results = []
    
    for category, image_files in categories.items():
        print(f"\n📊 Testing {category}:")
        print("-" * 40)
        
        category_adjectives = []
        category_captions = []
        
        for img_file in image_files:
            img_path = f"/data/coco/train2017/{img_file}"
            if not os.path.exists(img_path):
                continue
                
            try:
                image = Image.open(img_path)
                inputs = our_processor(images=image, return_tensors="pt").to(device)
                
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = our_model.generate(**inputs, max_length=50)
                
                caption = our_processor.decode(outputs[0], skip_special_tokens=True)
                adj_count = count_adjectives(caption)
                
                category_adjectives.append(adj_count)
                category_captions.append(caption)
                
                print(f"   🖼️ {img_file}: {adj_count} adjectives")
                print(f"      '{caption}'")
                
            except Exception as e:
                print(f"   ❌ Error with {img_file}: {e}")
                continue
        
        if category_adjectives:
            avg_adj = sum(category_adjectives) / len(category_adjectives)
            results.append({
                'category': category,
                'avg_adjectives': avg_adj,
                'samples': len(category_adjectives),
                'sample_caption': category_captions[0] if category_captions else ""
            })
            print(f"   📈 Category Average: {avg_adj:.2f} adjectives")
    
    # Generate category analysis
    print("\n" + "="*60)
    print("🏆 CATEGORY PERFORMANCE ANALYSIS")
    print("="*60)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('avg_adjectives', ascending=False)
        
        print("\n📈 Performance by Category (Ranked):")
        for i, row in df.iterrows():
            print(f"{i+1}. {row['category']:15} | Avg Adjectives: {row['avg_adjectives']:5.2f} | "
                  f"Samples: {row['samples']}")
        
        print("\n🎨 Best Performing Categories:")
        best_cat = df.iloc[0]
        print(f"   🥇 {best_cat['category']}: {best_cat['avg_adjectives']:.2f} adjectives")
        print(f"   📝 Sample: '{best_cat['sample_caption']}'")
        
        print(f"\n📉 Most Challenging Categories:")
        worst_cat = df.iloc[-1]
        print(f"   📍 {worst_cat['category']}: {worst_cat['avg_adjectives']:.2f} adjectives")
        print(f"   📝 Sample: '{worst_cat['sample_caption']}'")
        
        # Save category results
        with open("benchmark_category_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Category results saved to: benchmark_category_results.json")

if __name__ == "__main__":
    import os
    benchmark_by_category()
