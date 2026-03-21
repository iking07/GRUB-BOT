import torch
import time
import math

try:
    import custom_attention
    HAS_CUSTOM = True
except ImportError:
    HAS_CUSTOM = False
    print("WARNING: custom_attention not installed/compiled. Run `python setup_ext.py build_ext --inplace` with NVCC.")

def benchmark_attention(batch_size=8, num_heads=12, seq_len=256, head_dim=64, iterations=100):
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot benchmark CUDA kernels.")
        return
        
    device = torch.device("cuda")
    
    # Generate random Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Warmup standard attention
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
    torch.cuda.synchronize()
    start_sdpa = time.perf_counter()
    for _ in range(iterations):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    time_sdpa = (time.perf_counter() - start_sdpa) / iterations * 1000 # ms
    
    print(f"Standard PyTorch SDPA: {time_sdpa:.4f} ms / forward")
    
    if HAS_CUSTOM:
        # Warmup custom attention
        for _ in range(10):
            _ = custom_attention.forward(q, k, v)
            
        torch.cuda.synchronize()
        start_custom = time.perf_counter()
        for _ in range(iterations):
            _ = custom_attention.forward(q, k, v)
        torch.cuda.synchronize()
        time_custom = (time.perf_counter() - start_custom) / iterations * 1000 # ms
        
        print(f"Custom Native Attention: {time_custom:.4f} ms / forward")
        print(f"Speedup vs Standard: {time_sdpa / time_custom:.2f}x")
    else:
        print("Skipping custom attention benchmark because it is not compiled.")

if __name__ == "__main__":
    print("Benchmarking Transformer Attention Mechanisms...")
    benchmark_attention()
