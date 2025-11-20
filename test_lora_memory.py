from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

print("🧠 TESTING LoRA MEMORY OPTIMIZATION")
print("=" * 50)

# Test on OPT-1.3B (largest working model)
model_name = "facebook/opt-1.3b"
print(f"🤖 Testing LoRA on: {model_name}")

# Load model without LoRA
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.cuda()
base_memory = torch.cuda.memory_allocated() / 1024**3
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Base model: {total_params:,} parameters")
print(f"💾 Base memory: {base_memory:.1f} GB")

# Apply LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

lora_model = get_peft_model(model, lora_config)
lora_memory = torch.cuda.memory_allocated() / 1024**3

# Calculate memory savings
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)

print(f"🎯 LoRA Results:")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")
print(f"   Memory with LoRA: {lora_memory:.1f} GB")
print(f"   Memory reduction: {((base_memory - lora_memory) / base_memory * 100):.1f}%")
print(f"   Parameter efficiency: {(trainable_params/total_params*100):.2f}%")

# Test training
print("🧪 Testing LoRA training...")
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

# Quick training step
inputs = torch.randint(0, 50000, (2, 128)).cuda()
outputs = lora_model(inputs, labels=inputs)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"✅ LoRA training successful! Loss: {loss.item():.4f}")
print("🚀 LoRA READY for 3B model scaling!")
