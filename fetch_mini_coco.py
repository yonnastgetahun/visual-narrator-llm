import os, shutil
from datasets import load_dataset

out = "/home/ubuntu/data/coco/train2017"
os.makedirs(out, exist_ok=True)

# small slice of train split
ds = load_dataset("coco_captions", "2017", split="train[:1000]")
print("Downloading ~1000 images...")
for i, row in enumerate(ds):
    # row['image'] is a PIL image (HF auto-downloads the actual JPEGs)
    fn = os.path.join(out, f"{i:012d}.jpg")
    row["image"].save(fn, quality=90)
print("✅ Wrote images to:", out)
