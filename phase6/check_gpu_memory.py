import torch
import subprocess

print("🖥️ GPU Memory Check for 3B Model")

# Check current GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_memory = torch.cuda.memory_reserved(0) / 1e9
    available_memory = gpu_memory - free_memory
    
    print(f"📊 GPU Memory: {gpu_memory:.1f} GB total")
    print(f"📊 Available: {available_memory:.1f} GB free")
    
    # 3B model memory estimate (fp16)
    model_estimate = 6.0  # GB for 3B model in fp16
    training_estimate = 12.0  # GB for training with gradients
    
    print(f"🧮 3B Model estimate: {model_estimate} GB (fp16)")
    print(f"🧮 Training estimate: {training_estimate} GB")
    
    if available_memory > training_estimate:
        print("✅ Sufficient memory for 3B training!")
    else:
        print("⚠️  Memory may be tight for 3B training")
else:
    print("❌ CUDA not available")
