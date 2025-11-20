from PIL import Image, ImageDraw, ImageFont
import os, random
os.makedirs("dummy_images", exist_ok=True)
for i in range(32):
    img = Image.new("RGB", (512, 512), (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    d = ImageDraw.Draw(img)
    d.text((20,20), f"dummy {i}", fill=(255,255,255))
    img.save(f"dummy_images/img_{i:03d}.jpg", quality=90)
print("✅ Wrote 32 images to dummy_images/")
