import os
import requests
from PIL import Image
import io
import json

def download_coco_subset(num_images=100, output_dir="/home/ubuntu/data/coco/train2017"):
    """Download COCO subset using direct URLs"""
    
    print("📥 Downloading COCO subset via direct URLs...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample COCO image URLs (we'll use a small subset for testing)
    # These are public COCO images from the web
    sample_urls = [
        "http://images.cocodataset.org/train2017/000000000009.jpg",
        "http://images.cocodataset.org/train2017/000000000025.jpg", 
        "http://images.cocodataset.org/train2017/000000000030.jpg",
        "http://images.cocodataset.org/train2017/000000000034.jpg",
        "http://images.cocodataset.org/train2017/000000000036.jpg",
        "http://images.cocodataset.org/train2017/000000000042.jpg",
        "http://images.cocodataset.org/train2017/000000000051.jpg",
        "http://images.cocodataset.org/train2017/000000000052.jpg",
        "http://images.cocodataset.org/train2017/000000000061.jpg",
        "http://images.cocodataset.org/train2017/000000000064.jpg",
        "http://images.cocodataset.org/train2017/000000000072.jpg",
        "http://images.cocodataset.org/train2017/000000000074.jpg",
        "http://images.cocodataset.org/train2017/000000000085.jpg",
        "http://images.cocodataset.org/train2017/000000000094.jpg",
        "http://images.cocodataset.org/train2017/000000000097.jpg",
        "http://images.cocodataset.org/train2017/000000000104.jpg",
        "http://images.cocodataset.org/train2017/000000000106.jpg",
        "http://images.cocodataset.org/train2017/000000000110.jpg",
        "http://images.cocodataset.org/train2017/000000000113.jpg",
        "http://images.cocodataset.org/train2017/000000000119.jpg"
    ]
    
    saved_count = 0
    failed_count = 0
    
    for i, url in enumerate(sample_urls[:num_images]):
        try:
            print(f"📸 Downloading {i+1}/{min(num_images, len(sample_urls))}: {url}")
            
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save image
            image_path = f"{output_dir}/coco_downloaded_{i:08d}.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Verify image can be opened
            img = Image.open(image_path)
            img.verify()
            
            saved_count += 1
            print(f"   ✅ Saved: {image_path}")
            
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Failed: {e}")
            continue
    
    print(f"✅ Download complete! {saved_count} images saved, {failed_count} failed")
    return saved_count

if __name__ == "__main__":
    download_coco_subset(20)  # Start with 20 images for testing
