import os
import json
import glob

def check_current_status():
    """Check current dataset status and proceed accordingly"""
    
    print("🔍 Checking current training status...")
    
    # Check existing datasets
    datasets = {
        "original": "phase7/synth_train.json",
        "real": "phase7/synth_train_real.json"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"📁 {name}: {len(data)} samples")
            if len(data) > 0:
                print(f"   📝 Sample: {data[0]['caption'][:100]}...")
        else:
            print(f"📁 {name}: NOT FOUND")
    
    # Check images
    image_dirs = [
        "/data/coco/train2017",
        "/home/ubuntu/data/coco/train2017",
        "dummy_images"
    ]
    
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            images = glob.glob(f"{dir_path}/*.jpg") + glob.glob(f"{dir_path}/*.png")
            print(f"🖼️  {dir_path}: {len(images)} images")
        else:
            print(f"🖼️  {dir_path}: NOT FOUND")
    
    # Check checkpoints
    checkpoints = glob.glob("outputs/phase7_blip_synth_fp16/checkpoint-*")
    print(f"💾 Checkpoints: {len(checkpoints)} found")
    
    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    if len(checkpoints) > 0 and os.path.exists("phase7/synth_train.json"):
        print("✅ Ready to continue training with existing data")
        return "continue"
    elif os.path.exists("phase7/synth_train.json"):
        print("✅ Use existing synth_train.json for training")
        return "use_existing"
    else:
        print("❌ Need to create dataset first")
        return "create_dataset"

if __name__ == "__main__":
    status = check_current_status()
    print(f"\n🚀 Next step: {status}")
