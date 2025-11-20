import os
import json
import glob
import random
from pathlib import Path

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive']
    return sum(1 for adj in adjectives if adj in text.lower())

def create_scaled_dataset():
    """Create scaled dataset for Phase 7.3"""
    
    print("🔄 Creating Phase 7.3 scaled dataset...")
    
    # Use the downloaded 5K images
    image_dir = "/home/ubuntu/data/coco/train2017_5k"
    images = glob.glob(f"{image_dir}/*.jpg")
    
    # Also include existing images for more diversity
    existing_images = glob.glob("/data/coco/train2017/*.jpg") + glob.glob("/home/ubuntu/data/coco/train2017/*.jpg")
    all_images = list(set(images + existing_images))
    
    print(f"📊 Found {len(all_images)} total images for dataset creation")
    
    # Create dataset with high adjective density
    dataset = []
    adjective_pool = [
        "vivid", "gleaming", "rugged", "tranquil", "velvety", "golden",
        "cinematic", "dramatic", "vibrant", "serene", "majestic", "luminous",
        "textured", "atmospheric", "expressive", "stunning", "breathtaking"
    ]
    
    scene_types = [
        "landscape", "portrait", "street scene", "urban environment", 
        "natural setting", "indoor space", "architectural detail",
        "still life", "action shot", "candid moment"
    ]
    
    for img_path in all_images:
        # Create 3 augmented versions per image
        for aug_version in range(3):
            num_adjectives = random.randint(3, 6)
            selected_adjs = random.sample(adjective_pool, num_adjectives)
            scene_type = random.choice(scene_types)
            
            # Create descriptive caption
            filename = Path(img_path).stem
            if 'coco' in filename:
                subject = "scene"
            else:
                subject = filename.replace('_', ' ').replace('-', ' ')
            
            caption = f"a {', '.join(selected_adjs)} {scene_type} photograph of {subject}"
            
            dataset.append({
                "image": img_path,
                "caption": caption,
                "adjective_count": count_adjectives(caption),
                "augmentation": aug_version
            })
    
    # Save dataset
    output_path = "phase7/phase7_3_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Statistics
    total_adjectives = sum(item["adjective_count"] for item in dataset)
    avg_adjectives = total_adjectives / len(dataset)
    
    print(f"✅ Phase 7.3 dataset created: {output_path}")
    print(f"📊 Dataset Stats:")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Unique images: {len(all_images)}")
    print(f"   - Average adjectives: {avg_adjectives:.2f}")
    print(f"   - Augmentation factor: 3x")
    print(f"   - Expected training steps: {len(dataset) // 8}")
    
    # Show sample captions
    print(f"📝 Sample captions:")
    for i in range(min(3, len(dataset))):
        print(f"   {i+1}: {dataset[i]['caption']}")
    
    return dataset

if __name__ == "__main__":
    create_scaled_dataset()
