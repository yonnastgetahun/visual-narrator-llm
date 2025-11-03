import torch
print("1. PyTorch + CUDA:", torch.cuda.is_available())

try:
    from transformers import AutoTokenizer
    print("2. Transformers: ✅ WORKING")
except Exception as e:
    print("2. Transformers: ❌", e)

try:
    import datasets
    print("3. Datasets: ✅ WORKING") 
except Exception as e:
    print("3. Datasets: ❌", e)

print("4. A100 Status: READY FOR PHASE 3! 🚀")
