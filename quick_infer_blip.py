import os, re, sys
from PIL import Image, ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

ckpt = os.environ.get("PHASE7_CKPT", "outputs/phase7_blip/checkpoint-1762312409")
img_path = os.environ.get("PHASE7_TEST_IMAGE", "dummy_images/img_000.jpg")

print(f"Loading ckpt: {ckpt}")
processor = BlipProcessor.from_pretrained(ckpt, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(ckpt, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

if not os.path.isfile(img_path):
    # draw a quick synthetic image if not present
    img = Image.new("RGB", (512, 512), (30, 60, 120))
    d = ImageDraw.Draw(img); d.text((20,20), "sanity", fill=(255,255,255))
    img.save(img_path)

img = Image.open(img_path).convert("RGB")
inputs = processor(images=img, text="Describe the image", return_tensors="pt").to(model.device)
with torch.no_grad():
    out_ids = model.generate(**inputs, max_length=64)
caption = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
print("Caption:", caption)

# Simple adjective counter:
# 1) try NLTK POS-tagger if available; else heuristic suffix-based fallback
def count_adjs(text):
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        # if punkt not present, this will raise and we'll use fallback
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        return sum(1 for _, t in tags if t.startswith("JJ"))
    except Exception:
        # heuristic: words ending with common adjective suffixes
        suf = ("y","ful","ous","ive","al","ic","less","able","ible","ish","like","esque")
        return sum(1 for w in re.findall(r"[A-Za-z]+", text) if w.lower().endswith(suf))
print("Adj count:", count_adjs(caption))
