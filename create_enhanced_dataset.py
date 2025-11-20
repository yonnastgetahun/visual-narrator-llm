import os
import json
import glob
import random
from pathlib import Path

def count_adjectives(text):
    """Count adjectives in text"""
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive',
                 'stunning', 'breathtaking', 'captivating', 'mesmerizing']
    return sum(1 for adj in adjectives if adj in text.lower())

def adjective_augment(caption, adjective_ratio=0.9, min_adjs=2, max_adjs=5):
    """Augment caption with adjectives"""
    
    adjectives_pool = [
        "vivid", "gleaming", "rugged", "tranquil", "velvety", "golden", 
        "richly detailed", "cinematic", "dramatic", "vibrant", "serene",
        "majestic", "luminous", "textured", "atmospheric", "expressive",
        "stunning", "breathtaking", "captivating", "mesmerizing"
    ]
    
    if random.random() < adjective_ratio:
        num_adjectives = random.randint(min_adjs, max_adjs)
        selected_adjs = random.sample(adjectives_pool, num_adjectives)
        
        # Insert adjectives at beginning
        augmented = f"{', '.join(selected_adjs)} {caption}"
        return augmented
    
    return caption

def create_enhanced_dataset():
    """Create enhanced dataset from all available images"""
    
    # Find all images across different directories
    image_dirs = [
        "/data/coco/train2017",
        "/home/ubuntu/data/coco/train2017",
        "dummy_images"
    ]
    
    all_images = []
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            jpg_files = glob.glob(f"{dir_path}/*.jpg")
            png_files = glob.glob(f"{dir_path}/*.png")
            all_images.extend(jpg_files)
            all_images.extend(png_files)
    
    print(f"📊 Found {len(all_images)} total images")
    
    # Create enhanced dataset with high adjective density
    dataset = []
    
    for img_path in all_images:
        # Base caption from filename
        base_caption = f"a photo of {Path(img_path).stem.replace('_', ' ').replace('-', ' ')}"
        
        # Augment with high adjective density
        augmented_caption = adjective_augment(
            base_caption, 
            adjective_ratio=0.9,  # 90% get adjectives
            min_adjs=3,           # Minimum 3 adjectives
            max_adjs=5            # Maximum 5 adjectives
        )
        
        dataset.append({
            "image": img_path,
            "caption": augmented_caption,
            "adjective_count": count_adjectives(augmented_caption)
        })
    
    # Save dataset
    output_path = "phase7/synth_train_enhanced.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print stats
    total_images = len(dataset)
    total_adjectives = sum(item["adjective_count"] for item in dataset)
    avg_adjectives = total_adjectives / total_images if total_images > 0 else 0
    
    print(f"✅ Enhanced dataset created: {output_path}")
    print(f"📊 Dataset Stats:")
    print(f"   - Total samples: {total_images}")
    print(f"   - Total adjectives: {total_adjectives}")
    print(f"   - Average adjectives per caption: {avg_adjectives:.2f}")
    print(f"   - Target density: ≥3.0 adjectives/description")
    
    # Show sample captions
    print(f"📝 Sample captions:")
    for i in range(min(3, len(dataset))):
        print(f"   {i+1}: {dataset[i]['caption']}")
    
    return dataset

if __name__ == "__main__":
    create_enhanced_dataset()
