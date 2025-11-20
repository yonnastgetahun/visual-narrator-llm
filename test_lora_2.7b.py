from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

print("🚀 TESTING LoRA ON 2.7B MODEL")
print("=" * 50)

# Try OPT-2.7B
model_name = "facebook/opt-2.7b"
print(f"🤖 Testing: {model_name}")

try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Base parameters: {base_params:,}")
    
    # Apply LoRA before moving to GPU (saves memory)
    lora_config = LoraConfig(
        r=8,  # Smaller rank for larger model
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    lora_model = get_peft_model(model, lora_config)
    
    # Move to GPU
    lora_model = lora_model.cuda()
    memory_used = torch.cuda.memory_allocated() / 1024**3
    
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"🎯 2.7B + LoRA Results:")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Parameter efficiency: {(trainable_params/base_params*100):.3f}%")
    print(f"   GPU Memory: {memory_used:.1f} GB")
    print(f"   A100 Usage: {memory_used/40.0*100:.1f}%")
    print("🚀 2.7B MODEL + LoRA SUCCESS!")
    
except Exception as e:
    print(f"❌ 2.7B failed: {e}")
    print("💡 Falling back to 1.3B for Phase 4.1")
