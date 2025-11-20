import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", getattr(torchvision, "__version__", None))
print("cuda runtime:", getattr(getattr(torch, "version", None), "cuda", None))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("total memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
