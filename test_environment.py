import torch
import transformers
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers version:", transformers.__version__)

# Test what models are available
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✅ GPT-2 tokenizer loaded")
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print("❌ GPT-2 failed:", e)

try:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✅ GPT-2 model loaded")
except Exception as e:
    print("❌ GPT-2 model failed:", e)

# Check if we have our Phase 7 model
import os
if os.path.exists("outputs/phase7_complete"):
    print("✅ Phase 7 model exists")
else:
    print("❌ Phase 7 model not found")
    
# Check spatial dataset
import json
try:
    with open("phase8/spatial_intensive_dataset.json", "r") as f:
        data = json.load(f)
    print(f"✅ Spatial dataset: {len(data)} examples")
except:
    print("❌ Spatial dataset not found")
