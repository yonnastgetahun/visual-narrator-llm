import os
import requests
from PIL import Image
import time

def download_coco_5k():
    """Download 5K COCO images for Phase 7.3"""
    
    print("📥 Downloading 5K COCO images for Phase 7.3...")
    
    output_dir = "/home/ubuntu/data/coco/train2017_5k"
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO image IDs from the actual dataset
    coco_ids = list(range(1, 5001))
    
    saved_count = 0
    failed_count = 0
    
    for i, img_id in enumerate(coco_ids):
        if saved_count >= 5000:
            break
            
        url = f"http://images.cocodataset.org/train2017/{img_id:012d}.jpg"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image_path = f"{output_dir}/coco_{img_id:012d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image
                try:
                    img = Image.open(image_path)
                    img.verify()
                    saved_count += 1
                    if saved_count % 100 == 0:
                        print(f"📸 Downloaded {saved_count}/5000 images")
                except:
                    os.remove(image_path)
                    failed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
        
        # Progress every 100 attempts
        if i % 100 == 0:
            print(f"🔄 Processed {i}/5000 IDs, saved: {saved_count}, failed: {failed_count}")
    
    print(f"✅ Download complete: {saved_count} images saved, {failed_count} failed")
    return saved_count

if __name__ == "__main__":
    download_coco_5k()
