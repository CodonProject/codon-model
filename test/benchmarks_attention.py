import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from codon.block.attention import MultiHeadAttention
from codon.model.cache import build_cache

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def get_gpu_memory_mb():
    """获取当前 GPU 已分配的最大显存 (MB)"""
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark_mode(mode_name, model_kwargs, batch_size=2, seq_len=4096, decode_steps=100, device="cuda"):
    """针对指定模式运行 Prefill 和 Decode Benchmark"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    hidden_size = 4096
    num_heads = 32

    try:
        model = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **model_kwargs
        ).to(device).eval()

        # --- 1. Prefill 阶段 ---
        x_prefill = torch.randn(batch_size, seq_len, hidden_size, device=device)
        cache = build_cache(model)

        # Warmup
        for _ in range(3):
            _ = model(x_prefill, past_key_value=cache)
            cache.reset()

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            _ = model(x_prefill, past_key_value=cache)
        end_event.record()
        torch.cuda.synchronize()
        
        prefill_time_ms = start_event.elapsed_time(end_event)
        prefill_mem_mb = get_gpu_memory_mb()

        # --- 2. Decode 阶段 ---
        x_decode = torch.randn(batch_size, 1, hidden_size, device=device)
        
        for _ in range(5):
            _ = model(x_decode, past_key_value=cache)

        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            for _ in range(decode_steps):
                _ = model(x_decode, past_key_value=cache)
        end_event.record()
        torch.cuda.synchronize()

        decode_total_time_ms = start_event.elapsed_time(end_event)
        decode_per_token_ms = decode_total_time_ms / decode_steps
        peak_mem_mb = get_gpu_memory_mb()

        return {
            "mode": mode_name,
            "prefill_latency_ms": prefill_time_ms,
            "decode_per_token_ms": decode_per_token_ms,
            "peak_mem_mb": peak_mem_mb,
            "error": None
        }

    except Exception as e:
        # CUDA 异常隔离：清理失败上下文
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return {
            "mode": mode_name,
            "prefill_latency_ms": -1.0,
            "decode_per_token_ms": -1.0,
            "peak_mem_mb": -1.0,
            "error": str(e)
        }


def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA 不可用，性能基准测试必须在 GPU 上运行。")
        return

    device = "cuda"
    batch_size = 2
    seq_len = 4096
    decode_steps = 100

    print("=" * 95)
    print(f"Attention Benchmarks: PyTorch Native vs Triton CUDA Kernels")
    print(f"Config: Batch Size = {batch_size}, Prefill SeqLen = {seq_len}, Decode Steps = {decode_steps}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Triton Status: {'Available (' + triton.__version__ + ')' if HAS_TRITON else 'Not Installed'}")
    print("=" * 95)

    configs = [
        ("Standard MHA (Native)", {"use_triton": False}),
        ("GQA (8 KV Heads, Native)", {"num_kv_heads": 8, "use_triton": False}),
        ("GQA (8 KV Heads, Triton)", {"num_kv_heads": 8, "use_triton": True}),
        ("MLA (Native)", {"q_lora_rank": 1536, "kv_lora_rank": 512, "rope_dim": 64, "use_triton": False}),
        ("MLA (Triton)", {"q_lora_rank": 1536, "kv_lora_rank": 512, "rope_dim": 64, "use_triton": True}),
        ("HCA FP16 (Native)", {"use_hca": True, "hca_latent_dim": 512, "hca_block_size": 128, "hca_fp4_storage": False, "use_triton": False}),
        ("HCA FP16 (Triton)", {"use_hca": True, "hca_latent_dim": 512, "hca_block_size": 128, "hca_fp4_storage": False, "use_triton": True}),
        ("HCA FP4 (Native)", {"use_hca": True, "hca_latent_dim": 512, "hca_block_size": 128, "hca_fp4_storage": True, "use_triton": False}),
        ("CSA TopK=128 (Native)", {"use_csa": True, "csa_compressed_dim": 512, "csa_block_size": 4, "csa_top_k": 128, "use_triton": False}),
        ("CSA TopK=128 (Triton)", {"use_csa": True, "csa_compressed_dim": 512, "csa_block_size": 4, "csa_top_k": 128, "use_triton": True}),
    ]

    results = []
    for mode_name, kwargs in configs:
        res = benchmark_mode(
            mode_name=mode_name,
            model_kwargs=kwargs,
            batch_size=batch_size,
            seq_len=seq_len,
            decode_steps=decode_steps,
            device=device
        )
        results.append(res)
        status = "Failed" if res["error"] else "Finished"
        print(f"[{status:<8}] {mode_name}")

    print("\n" + "=" * 95)
    print(f"{'Attention Mode':<32} | {'Prefill (ms)':<14} | {'Decode (ms/tok)':<16} | {'Peak Mem (MB)':<12}")
    print("-" * 95)
    
    native_cache = {}
    for r in results:
        mode_str = r['mode']
        if r['error']:
            print(f"{mode_str:<32} | {'N/A':<14} | {'N/A':<16} | {'N/A':<12}")
            continue

        prefill_str = f"{r['prefill_latency_ms']:.2f}"
        decode_str = f"{r['decode_per_token_ms']:.3f}"
        mem_str = f"{r['peak_mem_mb']:.1f}"

        if "Native" in mode_str:
            base_key = mode_str.replace(" (Native)", "").replace(" Native", "")
            native_cache[base_key] = r
        elif "Triton" in mode_str:
            base_key = mode_str.replace(" (Triton)", "").replace(" Triton", "")
            if base_key in native_cache:
                nat_res = native_cache[base_key]
                p_speedup = nat_res['prefill_latency_ms'] / r['prefill_latency_ms'] if r['prefill_latency_ms'] > 0 else 1.0
                d_speedup = nat_res['decode_per_token_ms'] / r['decode_per_token_ms'] if r['decode_per_token_ms'] > 0 else 1.0
                
                prefill_str += f" ({p_speedup:.1f}x)"
                decode_str += f" ({d_speedup:.1f}x)"

        print(f"{mode_str:<32} | {prefill_str:<14} | {decode_str:<16} | {mem_str:<12}")
    
    print("=" * 95)


if __name__ == "__main__":
    run_benchmarks()