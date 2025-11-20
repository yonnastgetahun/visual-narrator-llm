"""
GH200 quick GPU sanity + perf benchmark for PyTorch.

What it measures (per dtype: fp32, fp16, bf16 when supported):
- 8192x8192 GEMM (matmul) throughput (approx TFLOPs)
- 64x64 conv2d (memory+compute mix) throughput
- Elementwise add on a large tensor (memory-bandwidth-ish proxy, GB/s)
- Tiny attention block (QK^T + softmax + V) latency

It also prints:
- Device name, capability, total memory
- Peak allocated memory during tests
"""

import time
import math
import torch
from contextlib import nullcontext

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # allow TF32 on Hopper for fp32 matmul speedups

def supports_fp16(device):
    return True  # all recent NVIDIA GPUs do; keep simple

def supports_bf16(device):
    # Hopper supports bfloat16; be safe:
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    return (cc_major, cc_minor) >= (8, 0)

def fmt(x, unit="s"):
    if unit == "s":
        if x < 1e-6: return f"{x*1e9:.2f} ns"
        if x < 1e-3: return f"{x*1e6:.2f} µs"
        if x < 1:    return f"{x*1e3:.2f} ms"
        return f"{x:.3f} s"
    return f"{x}"

def time_cuda(fn, iters=10, warmup=5):
    # Precise timing with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    # measure
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end)  # total ms for `iters`
    return (ms / iters) / 1000.0  # seconds per iter

def matmul_bench(dtype, device="cuda"):
    # Use square GEMM large enough to exercise tensor cores; keep reasonable for runtime
    N = 8192 if dtype in (torch.float16, torch.bfloat16) else 4096
    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)

    def run():
        c = a @ b
        # prevent DCE (dead code elimination)
        c = c[0, 0].item()

    sec = time_cuda(run, iters=5 if N==8192 else 8, warmup=3)
    # FLOPs for GEMM: 2*N^3
    tflops = (2 * (N**3)) / sec / 1e12
    return {"size": N, "sec": sec, "tflops": tflops}

def conv2d_bench(dtype, device="cuda"):
    # Typical CNN-ish: N=64, C=64, H=W=256, K=128, kernel 3x3, stride 1, padding 1
    N, C, H, W = 64, 64, 256, 256
    K = 128
    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    w = torch.randn(K, C, 3, 3, device=device, dtype=dtype)
    conv = torch.nn.Conv2d(C, K, 3, padding=1, bias=False, device=device, dtype=dtype)
    conv.weight.data.copy_(w)

    def run():
        y = conv(x)
        torch.cuda.synchronize()
        _ = y.mean().item()

    sec = time_cuda(run, iters=10, warmup=3)
    # Rough ops count: N*H*W*K*C*Kh*Kw*2 (mac -> 2 ops)
    ops = N * H * W * K * C * 3 * 3 * 2
    tflops = ops / sec / 1e12
    return {"sec": sec, "tflops": tflops}

def elemwise_bandwidth_bench(dtype, device="cuda"):
    # Large tensor add: memory dominated; gives a GB/s proxy
    # Use ~4 GiB per tensor if memory allows; GH200 has huge mem, but keep modest to finish quickly.
    # Let’s do ~1 GiB tensors per operand: 1e9 bytes / element_size.
    bytes_target = 1_000_000_000
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = bytes_target // elem_size
    a = torch.randn(numel, device=device, dtype=dtype)
    b = torch.randn(numel, device=device, dtype=dtype)

    def run():
        c = a + b
        torch.cuda.synchronize()
        _ = c[-1].item()

    sec = time_cuda(run, iters=20, warmup=5)
    # Read a + read b + write c ~= 3 * bytes_target per iter
    gbps = (3 * bytes_target) / sec / (1024**3)
    return {"sec": sec, "gbps": gbps, "numel": numel, "bytes_per_tensor": bytes_target}

def tiny_attention_bench(dtype, device="cuda"):
    # Single-head attention: Q,K,V in [B, T, D]
    B, T, D = 8, 2048, 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, T, D, device=device, dtype=dtype)
    k = torch.randn(B, T, D, device=device, dtype=dtype)
    v = torch.randn(B, T, D, device=device, dtype=dtype)

    def run():
        # attn weights: (B,T,T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        torch.cuda.synchronize()
        _ = out.mean().item()

    sec = time_cuda(run, iters=10, warmup=3)
    return {"sec": sec, "latency_ms": sec * 1e3}

def dtype_label(d):
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[d]

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    dev_idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev_idx)
    cc_major, cc_minor = torch.cuda.get_device_capability(dev_idx)
    total_mem = torch.cuda.get_device_properties(dev_idx).total_memory  # bytes

    print("="*80)
    print("PyTorch:", torch.__version__)
    print(f"Device: {name}")
    print(f"Compute Capability: {cc_major}.{cc_minor}")
    print(f"Total Memory: {total_mem/1024**3:.1f} GiB")
    print("="*80)

    # Warmup
    torch.manual_seed(0)
    x = torch.randn(1024, 1024, device=device)
    for _ in range(5):
        _ = x @ x
    torch.cuda.synchronize()

    # Track peak mem
    torch.cuda.reset_peak_memory_stats(device)

    dtypes = [torch.float32]
    if supports_fp16(device): dtypes.append(torch.float16)
    if supports_bf16(device): dtypes.append(torch.bfloat16)

    results = {}

    for dt in dtypes:
        print(f"\n--- Running benchmarks ({dtype_label(dt)}) ---")
        # autocast helps fp16/bf16 where relevant; matmul uses the tensor dtype directly
        ctx = nullcontext()
        if dt in (torch.float16, torch.bfloat16):
            ctx = torch.autocast(device_type="cuda", dtype=dt)

        with ctx:
            mm = matmul_bench(dt, device=device)
            print(f"Matmul {mm['size']}x{mm['size']}: {mm['tflops']:.2f} TFLOPs, {fmt(mm['sec'])}/iter")

            conv = conv2d_bench(dt, device=device)
            print(f"Conv2d: {conv['tflops']:.2f} TFLOPs, {fmt(conv['sec'])}/iter")

            bw = elemwise_bandwidth_bench(dt, device=device)
            print(f"Elemwise add (~mem BW): {bw['gbps']:.1f} GB/s, {fmt(bw['sec'])}/iter")

            attn = tiny_attention_bench(dt, device=device)
            print(f"Tiny attention latency: {attn['latency_ms']:.2f} ms/iter")

            results[dtype_label(dt)] = {
                "matmul_tflops": mm["tflops"],
                "conv_tflops": conv["tflops"],
                "mem_gbps": bw["gbps"],
                "attn_ms": attn["latency_ms"],
            }

    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"\nPeak allocated during run: {peak:.2f} GiB")

    print("\n=== Summary ===")
    for k, v in results.items():
        print(
            f"{k:>5} | GEMM: {v['matmul_tflops']:.2f} TFLOPs | Conv: {v['conv_tflops']:.2f} TFLOPs | "
            f"BW-proxy: {v['mem_gbps']:.1f} GB/s | Attn: {v['attn_ms']:.2f} ms"
        )

    print("\nNotes:")
    print("- GEMM uses 8192^2 for fp16/bf16 and 4096^2 for fp32 to keep runtime reasonable.")
    print("- BW figure is an approximation (elementwise add ~ 3x tensor bytes per iter).")
    print("- TF32 may accelerate fp32 matmul on Hopper; disable via: torch.set_float32_matmul_precision('high'|'medium'|'highest').")
    print("- For stable numbers, run when the machine is otherwise idle.")

if __name__ == "__main__":
    main()
