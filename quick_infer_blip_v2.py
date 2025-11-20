import os, re
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

ckpt = os.environ.get("PHASE7_CKPT")
if not ckpt:
    # fallback to the latest synth ckpt if present, else the earlier single-step ckpt
    import glob
    synth = sorted(glob.glob("outputs/phase7_blip_synth/checkpoint-*")) or []
    ckpt = synth[-1] if synth else "outputs/phase7_blip/checkpoint-1762312576"

imgp = os.environ.get("PHASE7_TEST_IMAGE", "dummy_images/img_000.jpg")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(ckpt)
model     = BlipForConditionalGeneration.from_pretrained(
    ckpt, torch_dtype=(torch.bfloat16 if device=="cuda" else None)
).to(device)

image = Image.open(imgp).convert("RGB")
prompt = "a richly detailed, vivid, cinematic photo of"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=48,
        num_beams=5,
        length_penalty=0.9,
        repetition_penalty=1.2,
        early_stopping=True
    )

text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)

adjectives = re.findall(r"\b(beautiful|vivid|rich|textured|moody|dramatic|glossy|matte|soft|sharp|vibrant|dusky|lush|crisp|noisy|gritty|elegant|somber|warm|cool|glowing|gleaming|faded|weathered|ornate|minimal)\b", text.lower())
print("CKPT:", ckpt)
print("Caption:", text)
print("Adj count:", len(adjectives))
