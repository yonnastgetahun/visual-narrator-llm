import json
import os

def diagnose_dataset():
    """Diagnose the dataset issues"""
    
    dataset_path = "phase7/synth_train_enhanced.json"
    
    if not os.path.exists(dataset_path):
        print("❌ Enhanced dataset not found!")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print("🔍 DATASET DIAGNOSIS")
    print("=" * 60)
    print(f"📊 Total samples: {len(data)}")
    
    # Check if images exist
    existing_images = 0
    missing_images = 0
    
    for i, item in enumerate(data):
        if os.path.exists(item["image"]):
            existing_images += 1
        else:
            missing_images += 1
            if missing_images <= 3:  # Show first 3 missing
                print(f"❌ Missing image: {item['image']}")
    
    print(f"🖼️  Images: {existing_images} exist, {missing_images} missing")
    
    # Check adjective density
    total_adjectives = sum(item.get("adjective_count", 0) for item in data)
    avg_adjectives = total_adjectives / len(data) if data else 0
    
    print(f"🎯 Adjective density: {avg_adjectives:.2f} (target: ≥3.0)")
    
    # Show sample captions
    print(f"📝 Sample captions (first 5):")
    for i in range(min(5, len(data))):
        adj_count = data[i].get("adjective_count", 0)
        print(f"   {i+1}. [{adj_count} adj] {data[i]['caption']}")
    
    # Calculate expected training steps
    batch_size = 8
    expected_steps = len(data) // batch_size
    print(f"📈 Expected training: {expected_steps} steps (batch_size={batch_size})")
    
    return len(data), existing_images, avg_adjectives

if __name__ == "__main__":
    diagnose_dataset()
