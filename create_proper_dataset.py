import os
import json
import glob
import random
from pathlib import Path

def count_adjectives(text):
    adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                 'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
                 'majestic', 'luminous', 'textured', 'atmospheric', 'expressive',
                 'stunning', 'breathtaking', 'captivating', 'mesmerizing']
    return sum(1 for adj in adjectives if adj in text.lower())

def create_proper_dataset():
    """Create a proper training dataset with all available images"""
    
    print("🔄 Creating proper training dataset...")
    
    # Find ALL images in the system
    search_paths = [
        "/data/coco/train2017",
        "/home/ubuntu/data/coco/train2017", 
        "dummy_images",
        "/home/ubuntu/visual-narrator-llm"
    ]
    
    all_images = []
    for path in search_paths:
        if os.path.exists(path):
            jpg_files = glob.glob(f"{path}/*.jpg")
            png_files = glob.glob(f"{path}/*.png")
            all_images.extend(jpg_files)
            all_images.extend(png_files)
            print(f"📁 Found {len(jpg_files) + len(png_files)} images in {path}")
    
    # Remove duplicates by keeping only unique filenames
    unique_images = []
    seen_names = set()
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        if img_name not in seen_names and os.path.exists(img_path):
            unique_images.append(img_path)
            seen_names.add(img_name)
    
    print(f"📊 Total unique images: {len(unique_images)}")
    
    if len(unique_images) < 50:
        print("⚠️  Warning: Dataset is still small. We'll use data augmentation.")
    
    # Create dataset with high adjective density
    dataset = []
    adjective_pool = [
        "vivid", "gleaming", "rugged", "tranquil", "velvety", "golden",
        "cinematic", "dramatic", "vibrant", "serene", "majestic", "luminous",
        "textured", "atmospheric", "expressive", "stunning", "breathtaking",
        "captivating", "mesmerizing", "radiant", "glowing", "sparkling"
    ]
    
    scene_types = [
        "landscape", "portrait", "street scene", "urban environment", 
        "natural setting", "indoor space", "architectural detail",
        "still life", "action shot", "candid moment"
    ]
    
    for img_path in unique_images:
        # Create multiple augmented versions of each image
        for aug_version in range(3):  # 3 versions per image
            num_adjectives = random.randint(3, 6)  # 3-6 adjectives
            selected_adjs = random.sample(adjective_pool, num_adjectives)
            scene_type = random.choice(scene_types)
            
            caption = f"a {', '.join(selected_adjs)} {scene_type} photograph of {Path(img_path).stem.replace('_', ' ')}"
            
            dataset.append({
                "image": img_path,
                "caption": caption,
                "adjective_count": count_adjectives(caption),
                "augmentation": aug_version
            })
    
    print(f"📈 Created {len(dataset)} training samples (augmented)")
    
    # Save dataset
    output_path = "phase7/synth_train_proper.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Calculate statistics
    total_adjectives = sum(item["adjective_count"] for item in dataset)
    avg_adjectives = total_adjectives / len(dataset)
    
    print("✅ PROPER DATASET CREATED")
    print(f"📊 Final Stats:")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Unique images: {len(unique_images)}")
    print(f"   - Average adjectives: {avg_adjectives:.2f} per caption")
    print(f"   - Augmentation factor: 3x")
    print(f"   - Expected training steps: {len(dataset) // 8} (batch_size=8)")
    
    # Verify some samples exist
    print(f"🔍 Verifying image accessibility...")
    accessible = 0
    for item in dataset[:10]:  # Check first 10
        if os.path.exists(item["image"]):
            accessible += 1
    
    print(f"   - Images accessible: {accessible}/10 checked")
    
    return dataset

if __name__ == "__main__":
    create_proper_dataset()
