import torch, math

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"

dt = torch.bfloat16
print("bf16 element_size:", torch.tensor([], dtype=dt).element_size(), "bytes")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # allow TF32 fast path

B, T, D = 8, 2048, 128
scale = 1.0 / math.sqrt(D)
with torch.autocast(device_type="cuda", dtype=dt):
    q = torch.randn(B, T, D, device="cuda")
    k = torch.randn(B, T, D, device="cuda")
    v = torch.randn(B, T, D, device="cuda")
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    torch.cuda.synchronize()
print("✅ BF16 attention ran OK. mean:", out.mean().item())
