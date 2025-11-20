from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

print("🧠 OFFICIAL PEFT LoRA TEST")
print("=" * 50)

# Test on OPT-1.3B
model_name = "facebook/opt-1.3b"
print(f"🤖 Testing LoRA on: {model_name}")

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
base_params = sum(p.numel() for p in model.parameters())
print(f"📊 Base parameters: {base_params:,}")

# Move to GPU and measure memory
model = model.cuda()
base_memory = torch.cuda.memory_allocated() / 1024**3
print(f"💾 Base memory: {base_memory:.1f} GB")

# Apply LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # OPT attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

lora_model = get_peft_model(model, lora_config)
lora_memory = torch.cuda.memory_allocated() / 1024**3

# Get trainable parameters
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)

print(f"🎯 LoRA Results:")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")
print(f"   Total parameters: {trainable_params + frozen_params:,}")
print(f"   Memory with LoRA: {lora_memory:.1f} GB")
print(f"   Parameter efficiency: {(trainable_params/base_params*100):.2f}%")

# Test training
print("🧪 Testing LoRA training...")
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

# Quick training step
inputs = torch.randint(0, 50000, (2, 64)).cuda()
outputs = lora_model(inputs, labels=inputs)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"✅ LoRA training successful! Loss: {loss.item():.4f}")

# Test inference
print("🧪 Testing inference...")
with torch.no_grad():
    generated = lora_model.generate(inputs, max_length=80)
print(f"✅ Inference successful! Generated length: {generated.shape[1]}")

print("🚀 OFFICIAL PEFT LoRA WORKING PERFECTLY!")
print("🎯 Ready for 3B model scaling!")
