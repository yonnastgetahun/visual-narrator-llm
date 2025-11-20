import sys, subprocess

pkgs = [
  "huggingface_hub>=0.26.2",
  "accelerate>=1.11.0",
  "transformers>=4.45.0",
  "datasets>=3.0.2",
  "evaluate",
  "nltk>=3.9.1",
  # (Optional) wandb for logs
  # "wandb"
]

print("➡️  Upgrading:", pkgs)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + pkgs)
print("✅ Dependencies aligned")
