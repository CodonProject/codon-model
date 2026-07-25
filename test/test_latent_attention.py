import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import unittest
import gc

from codon.exp.block.attn_latent import MultiHeadAttention
from codon.block.embedding import RotaryEmbedding


class TestAttentionCorrectness(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 512
        self.num_heads = 8
        self.head_dim = self.hidden_size // self.num_heads
        self.batch_size = 2
        self.seq_len = 16
        self.rope_dim = 32
        
        self.position_emb = RotaryEmbedding(model_dim=self.rope_dim, max_len=128)

    def _run_forward(self, **kwargs):
        model = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            **kwargs
        ).eval()
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        with torch.no_grad():
            out = model(x, position_emb=self.position_emb)
        return out.output.shape

    def test_forward_shapes(self):
        configs = [
            {},
            {'num_kv_heads': 2},
            {'q_lora_rank': 128, 'kv_lora_rank': 256, 'rope_dim': self.rope_dim},
            {'q_lora_rank': 128, 'rope_dim': self.rope_dim},
            {'kv_lora_rank': 256, 'rope_dim': self.rope_dim},
            {'use_qk_norm': False, 'use_gate': True}
        ]
        for cfg in configs:
            shape = self._run_forward(**cfg)
            self.assertEqual(shape, (self.batch_size, self.seq_len, self.hidden_size), 
                             f'Shape mismatch for config: {cfg}')

    def test_kv_cache_consistency(self):
        configs = [
            {'name': 'MHA', 'kwargs': {}},
            {'name': 'GQA', 'kwargs': {'num_kv_heads': 2}},
            {'name': 'MLA (Full)', 'kwargs': {'q_lora_rank': 128, 'kv_lora_rank': 256, 'rope_dim': self.rope_dim}},
            {'name': 'MLA (KV only)', 'kwargs': {'kv_lora_rank': 256, 'rope_dim': self.rope_dim}},
        ]
        
        for cfg in configs:
            model = MultiHeadAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                **cfg['kwargs']
            ).eval()
            
            x_full = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
            
            with torch.no_grad():
                out_full = model(x_full, position_emb=self.position_emb, use_cache=False).output
                
            prefill_len = self.seq_len // 2
            x_prefill = x_full[:, :prefill_len, :]
            x_decode = x_full[:, prefill_len:, :]
            
            with torch.no_grad():
                out_prefill = model(x_prefill, position_emb=self.position_emb, use_cache=True, embedding_start=0)
                past_kv = out_prefill.past_key_value
                
                out_decode = model(
                    x_decode, position_emb=self.position_emb, use_cache=True, 
                    past_key_value=past_kv, embedding_start=prefill_len
                ).output
                
            out_cached = torch.cat([out_prefill.output, out_decode], dim=1)
            
            diff = (out_full - out_cached).abs().max().item()
            self.assertLess(diff, 1e-4, f'KV Cache consistency failed for {cfg['name']}, max diff: {diff}')

    def test_backward_pass(self):
        model = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            kv_lora_rank=256,
            rope_dim=self.rope_dim
        ).train()
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        out = model(x, position_emb=self.position_emb).output
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad, 'Input gradient is None')
        self.assertTrue(x.grad.abs().sum() > 0, 'Input gradient is all zeros')


def benchmark_attention():
    if not torch.cuda.is_available():
        print('\n[CUDA not available, skipping GPU benchmark.]\n')
        return

    device = torch.device('cuda')
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads  # 128
    rope_dim = 64
    batch_size = 1
    prefill_len = 2048
    decode_steps = 100

    position_emb = RotaryEmbedding(model_dim=rope_dim, max_len=prefill_len + decode_steps).to(device)

    configs = {
        'MHA': {},
        'GQA': {'num_kv_heads': 8},
        'MLA (KV only)': {'kv_lora_rank': 512, 'rope_dim': rope_dim},
        'MLA (Q+KV)': {'q_lora_rank': 512, 'kv_lora_rank': 512, 'rope_dim': rope_dim},
    }

    print(f'\n{'='*70}')
    print(f'Attention Benchmark (Device: {torch.cuda.get_device_name(0)})')
    print(f'Hidden: {hidden_size}, Heads: {num_heads}, Head_Dim: {head_dim}, RoPE_Dim: {rope_dim}')
    print(f'Prefill Len: {prefill_len}, Decode Steps: {decode_steps}')
    print(f'{'='*70}\n')

    for name, kwargs in configs.items():
        model = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        ).to(device).eval()

        x_prefill = torch.randn(batch_size, prefill_len, hidden_size, device=device)
        x_decode = torch.randn(batch_size, 1, hidden_size, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(x_prefill, position_emb=position_emb)
        torch.cuda.synchronize()

        # Prefill Benchmark
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(10):
                out = model(x_prefill, position_emb=position_emb, use_cache=True)
        end_event.record()
        torch.cuda.synchronize()
        
        prefill_time = start_event.elapsed_time(end_event) / 10
        prefill_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Decode Benchmark
        past_kv = out.past_key_value
        torch.cuda.reset_peak_memory_stats()
        
        start_event.record()
        with torch.no_grad():
            for step in range(decode_steps):
                out = model(x_decode, position_emb=position_emb, past_key_value=past_kv, 
                            use_cache=True, embedding_start=prefill_len + step)
                past_kv = out.past_key_value
        end_event.record()
        torch.cuda.synchronize()
        
        decode_time = start_event.elapsed_time(end_event) / decode_steps
        decode_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f'[{name}]')
        print(f'  Prefill Latency : {prefill_time:.2f} ms')
        print(f'  KV Cache Memory : {prefill_mem:.2f} MB')
        print(f'  Decode Latency  : {decode_time:.2f} ms/step')
        print(f'  Decode Memory   : {decode_mem:.2f} MB')
        print('-' * 70)
        
        del model, x_prefill, x_decode, out, past_kv
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print('Running Correctness Tests...')
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    print('\nRunning Performance Benchmark...')
    benchmark_attention()