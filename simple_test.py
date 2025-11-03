import torch
print("1. Testing PyTorch + CUDA...")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")

try:
    from transformers import __version__ as tf_version
    print(f"2. Transformers: {tf_version} ✅")
except ImportError as e:
    print(f"2. Transformers: Not installed ❌")
    print(f"   Error: {e}")

try:
    import datasets
    print(f"3. Datasets: {datasets.__version__} ✅")
except ImportError as e:
    print(f"3. Datasets: Not installed ❌")

print("4. A100 Status: READY" if torch.cuda.is_available() else "A100 Status: ISSUE")
