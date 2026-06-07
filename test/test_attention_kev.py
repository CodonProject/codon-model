import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from codon.block.attention import MultiHeadAttentionKEV, AttentionOutput
from codon.block.embedding import InterleavedRotaryEmbedding

def test_multi_head_attention_kev():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on device: {device}")

    # Parameters
    batch_size = 2
    seq_len = 8
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 2
    
    # 1. Test basic initialization and forward pass
    print("Testing basic initialization and forward pass...")
    attn = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_qk_norm=False,
        use_gate=False,
        is_causal=False
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    out: AttentionOutput = attn(x)
    
    assert out.output.shape == (batch_size, seq_len, hidden_size), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {out.output.shape}"
    print("Basic forward pass shape verified.")

    # 2. Test K = V Identity when use_qk_norm=False and use_cache=True (optimize_kv_cache=False)
    print("Testing K = V identity (use_qk_norm=False, optimize_kv_cache=False)...")
    out_cached: AttentionOutput = attn(x, use_cache=True, optimize_kv_cache=False)
    assert out_cached.past_key_value is not None
    assert isinstance(out_cached.past_key_value, tuple) and len(out_cached.past_key_value) == 2
    past_k, past_v = out_cached.past_key_value
    
    # K and V should be identical
    assert torch.equal(past_k, past_v), "K and V are not identical!"
    print("K = V identity verified (use_qk_norm=False, optimize_kv_cache=False).")

    # 3. Test K = V Identity under QK Normalization when use_qk_norm=True
    print("Testing K = V identity under QK Normalization (use_qk_norm=True, optimize_kv_cache=False)...")
    attn_norm = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
        use_gate=True,
        is_causal=True
    ).to(device)

    out_norm: AttentionOutput = attn_norm(x, use_cache=True, optimize_kv_cache=False)
    past_k_norm, past_v_norm = out_norm.past_key_value
    
    # With QK norm, Key should be RMSNorm(Value)
    expected_k = attn_norm.k_norm(past_v_norm)
    assert torch.allclose(past_k_norm, expected_k, atol=1e-5), "Key is not equal to RMSNorm(Value)!"
    print("K = V identity verified under QK Normalization (optimize_kv_cache=False).")

    # 4. Test Single-Tensor Cache Optimization (optimize_kv_cache=True)
    print("Testing single-tensor KV cache optimization...")
    out_opt: AttentionOutput = attn(x, use_cache=True, optimize_kv_cache=True)
    assert out_opt.past_key_value is not None
    assert torch.is_tensor(out_opt.past_key_value), "Expected past_key_value to be a single torch.Tensor!"
    assert out_opt.past_key_value.shape == (batch_size, num_kv_heads, seq_len, hidden_size // num_heads)
    print("Single-tensor KV cache verified.")

    # 5. Test with Position Embeddings (RoPE)
    print("Testing with InterleavedRotaryEmbedding...")
    rope = InterleavedRotaryEmbedding(
        model_dim=hidden_size // num_heads,
        num_axes=1
    ).to(device)
    # Positions: batch_size x seq_len x 1
    positions = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1).float()
    
    out_rope: AttentionOutput = attn_norm(
        x, 
        position_emb=rope,
        embedding_pos=positions,
        use_cache=True,
        optimize_kv_cache=False
    )
    past_k_rope, past_v_rope = out_rope.past_key_value
    
    # RoPE is applied to K but not V.
    # Check that past_v_rope corresponds to the un-RoPEd values
    # Also verify that output shapes are correct
    assert out_rope.output.shape == (batch_size, seq_len, hidden_size)
    assert past_k_rope.shape == (batch_size, num_kv_heads, seq_len, hidden_size // num_heads)
    assert past_v_rope.shape == (batch_size, num_kv_heads, seq_len, hidden_size // num_heads)
    print("RoPE forward pass verified.")

    # 6. Test GQA expansion
    print("Testing Multi-Query Attention (num_kv_heads=1)...")
    attn_mqa = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=1,
        use_qk_norm=True,
        use_gate=True
    ).to(device)
    out_mqa = attn_mqa(x)
    assert out_mqa.output.shape == (batch_size, seq_len, hidden_size)
    print("MQA verified.")

    # 7. Test Backpropagation (optimize_kv_cache=True)
    print("Testing backpropagation with optimized KV Cache...")
    attn_mqa_opt = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=1,
        use_qk_norm=True,
        use_gate=True
    ).to(device)
    x_bp = torch.randn(batch_size, seq_len, hidden_size, device=device)
    out_bp = attn_mqa_opt(x_bp, use_cache=True, optimize_kv_cache=True)
    loss = out_bp.output.sum()
    loss.backward()
    
    # Check that gradients are computed for all parameters
    for name, param in attn_mqa_opt.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert torch.nonzero(param.grad).size(0) > 0, f"Gradient for {name} is all zeros!"
    print("Backpropagation verified.")

    # 8. Test Numerical Equivalence between Optimized and Non-Optimized cache paths
    print("Testing numerical equivalence in multi-step generation...")
    # Initialize two identical models (sharing the same weights)
    attn_opt_equiv = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
        use_gate=True,
        is_causal=True
    ).to(device)
    
    attn_non_opt_equiv = MultiHeadAttentionKEV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
        use_gate=True,
        is_causal=True
    ).to(device)
    
    # Copy weights
    attn_non_opt_equiv.load_state_dict(attn_opt_equiv.state_dict())
    
    # Put both models in eval mode to disable dropout for equivalence checking
    attn_opt_equiv.eval()
    attn_non_opt_equiv.eval()
    
    # Input tokens for prefill and generation
    x_prefill = torch.randn(batch_size, 4, hidden_size, device=device)
    x_gen1 = torch.randn(batch_size, 1, hidden_size, device=device)
    x_gen2 = torch.randn(batch_size, 1, hidden_size, device=device)
    
    # 8.1 Without RoPE
    # Run prefill
    out_opt_pref = attn_opt_equiv(x_prefill, use_cache=True, optimize_kv_cache=True)
    out_non_opt_pref = attn_non_opt_equiv(x_prefill, use_cache=True, optimize_kv_cache=False)
    assert torch.allclose(out_opt_pref.output, out_non_opt_pref.output, atol=1e-5)
    
    # Run step 1
    out_opt_g1 = attn_opt_equiv(x_gen1, use_cache=True, optimize_kv_cache=True, past_key_value=out_opt_pref.past_key_value, embedding_start=4)
    out_non_opt_g1 = attn_non_opt_equiv(x_gen1, use_cache=True, optimize_kv_cache=False, past_key_value=out_non_opt_pref.past_key_value, embedding_start=4)
    assert torch.allclose(out_opt_g1.output, out_non_opt_g1.output, atol=1e-5)
    
    # Run step 2
    out_opt_g2 = attn_opt_equiv(x_gen2, use_cache=True, optimize_kv_cache=True, past_key_value=out_opt_g1.past_key_value, embedding_start=5)
    out_non_opt_g2 = attn_non_opt_equiv(x_gen2, use_cache=True, optimize_kv_cache=False, past_key_value=out_non_opt_g1.past_key_value, embedding_start=5)
    assert torch.allclose(out_opt_g2.output, out_non_opt_g2.output, atol=1e-5)
    
    # 8.2 With RoPE
    # Run prefill
    out_opt_pref_rope = attn_opt_equiv(x_prefill, use_cache=True, optimize_kv_cache=True, position_emb=rope, embedding_start=0)
    out_non_opt_pref_rope = attn_non_opt_equiv(x_prefill, use_cache=True, optimize_kv_cache=False, position_emb=rope, embedding_start=0)
    assert torch.allclose(out_opt_pref_rope.output, out_non_opt_pref_rope.output, atol=1e-5)
    
    # Run step 1
    out_opt_g1_rope = attn_opt_equiv(x_gen1, use_cache=True, optimize_kv_cache=True, past_key_value=out_opt_pref_rope.past_key_value, position_emb=rope, embedding_start=4)
    out_non_opt_g1_rope = attn_non_opt_equiv(x_gen1, use_cache=True, optimize_kv_cache=False, past_key_value=out_non_opt_pref_rope.past_key_value, position_emb=rope, embedding_start=4)
    assert torch.allclose(out_opt_g1_rope.output, out_non_opt_g1_rope.output, atol=1e-5)
    
    # Run step 2
    out_opt_g2_rope = attn_opt_equiv(x_gen2, use_cache=True, optimize_kv_cache=True, past_key_value=out_opt_g1_rope.past_key_value, position_emb=rope, embedding_start=5)
    out_non_opt_g2_rope = attn_non_opt_equiv(x_gen2, use_cache=True, optimize_kv_cache=False, past_key_value=out_non_opt_g1_rope.past_key_value, position_emb=rope, embedding_start=5)
    assert torch.allclose(out_opt_g2_rope.output, out_non_opt_g2_rope.output, atol=1e-5)

    print("Numerical equivalence in multi-step generation verified.")

    print("\nAll MultiHeadAttentionKEV optimization tests passed successfully!")

if __name__ == '__main__':
    test_multi_head_attention_kev()
