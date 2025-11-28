# hydragen_bench.py
import torch
import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydragen.attention import hydragen_attention

def benchmark_hydragen_baseline():
    print(f"{'='*60}")
    print(f"BENCHMARK: Hydragen w/ FlashAttention (Baseline)")
    print(f"{'='*60}")
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. Setup Data for a Prefill Scenario
    # Match TK benchmark exactly: Batch=16, Seq=768, Heads=16, HeadDim=128
    B, S, H, D = 16, 6144, 16, 128
    
    print(f"Configuration: B={B}, S={S}, H={H}, D={D}")
    
    # Create Tensors
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    
    # Dummy shared lists (we test the 'unique' path which is self-attention)
    # This is equivalent to prefilling the prompt.
    
    # 2. Warmup
    print("Warming up...")
    for _ in range(10):
        _ = hydragen_attention(
            q, k, v, 
            shared_ks=[], shared_vs=[], 
            shared_cu_seq_lens=[], shared_max_seq_lens=[], use_varlens=[], seq_lens=None
        )
    torch.cuda.synchronize()
    
    # 3. Benchmark
    iters = 100
    print(f"Running {iters} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = hydragen_attention(
            q, k, v, 
            shared_ks=[], shared_vs=[], 
            shared_cu_seq_lens=[], shared_max_seq_lens=[], use_varlens=[], seq_lens=None
        )
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    
    # 4. Calculate TFLOPS
    # FLOPs for Causal Attention = 4 * B * H * S^2 * D (forward pass)
    # Standard formula counting MACs as 2 operations
    flops = 4 * B * H * (S**2) * D
    tflops = (flops / (avg_ms / 1000)) / 1e12
    
    print(f"\nResults:")
    print(f"  Avg Latency: {avg_ms:.4f} ms")
    print(f"  Est. Throughput: {tflops:.2f} TFLOPS")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    benchmark_hydragen_baseline()