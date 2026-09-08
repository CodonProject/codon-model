from codon import *
from codon.block.attention.base import BasicLinearAttention
from codon.model.cache import GatedDeltaAttentionLayerCache
from codon.ops import AttentionOutput
from codon.ops.attn_cuda.gdn import (
    recurrent_gated_delta_rule,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_triton,
    HAS_TRITON as HAS_TRITON_GDN,
)


class GatedRMSNorm(BasicModel):
    '''RMSNorm(x) * silu(gate)，GatedDeltaNet 的输出归一化。'''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate=None):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (xf * self.weight.float()).to(dtype)
        if gate is not None:
            out = out * F.silu(gate)
        return out


class GatedDeltaAttention(BasicLinearAttention):
    '''
    Gated DeltaNet 注意力（Mamba2 风格 delta rule + 遗忘门 + 写入强度 + 输出门）。

    几何与 BasicLinearAttention 不同：key/query 头维 dk（默认 = head_dim），
    value 头维 dv = dk * expand_v（被放大），故绕过父类 reshape_q/reshape_kv，
    forward 全量重写，仅继承类型身份与 hidden_size/num_heads/head_dim 字段。
    '''
    def __init__(self, hidden_size, num_heads, head_k_dim=None, expand_v=2.0,
                 conv_size=4, chunk_size=64, bias=False, dropout=0.0, **kwargs):
        super().__init__(hidden_size=hidden_size, num_heads=num_heads,
                         num_kv_heads=num_heads, dropout=dropout, bias=bias, **kwargs)

        self.dk = head_k_dim or self.head_dim
        self.dv = int(self.dk * expand_v)
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        dk_all = self.num_heads * self.dk
        dv_all = self.num_heads * self.dv

        # 覆盖父类按 head_dim/kv_dim 建的投影（几何不符）
        self.q_proj = nn.Linear(hidden_size, dk_all, bias=bias)
        self.k_proj = nn.Linear(hidden_size, dk_all, bias=bias)
        self.v_proj = nn.Linear(hidden_size, dv_all, bias=bias)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=bias)   # 遗忘门 α
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=bias)   # 写入强度 β
        self.g_proj = nn.Linear(hidden_size, dv_all, bias=bias)           # 输出门
        self.o_proj = nn.Linear(dv_all, hidden_size, bias=bias)           # 输入是 dv_all

        # depthwise short conv（padding 手动处理以支持缓存）
        self.q_conv = nn.Conv1d(dk_all, dk_all, conv_size, groups=dk_all, bias=False)
        self.k_conv = nn.Conv1d(dk_all, dk_all, conv_size, groups=dk_all, bias=False)
        self.v_conv = nn.Conv1d(dv_all, dv_all, conv_size, groups=dv_all, bias=False)

        # Mamba2 风格参数化：α = exp(-A * softplus(a + dt_bias))
        A = torch.empty(self.num_heads).uniform_(1, 16)
        self.A_log = nn.Parameter(A.log())
        dt = torch.exp(torch.rand(self.num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm = GatedRMSNorm(self.dv)

    @classmethod
    def cache_type(cls):
        return GatedDeltaAttentionLayerCache

    def _conv(self, x, conv, cache):
        # x: (B, L, D) -> (B, L, D)，返回新的 conv 缓存 (B, D, conv_size-1)
        B, L, D = x.shape
        x = x.transpose(1, 2)
        if cache is None:
            cache = x.new_zeros(B, D, self.conv_size - 1)
        xc = torch.cat([cache, x], dim=-1)
        new_cache = xc[..., -(self.conv_size - 1):]
        return F.silu(conv(xc)).transpose(1, 2), new_cache

    def forward(self, hidden_states, attention_mask=None, output_attentions=False,
                position_emb=None, embedding_start=0, embedding_pos=None,
                past_key_value=None) -> AttentionOutput:
        B, L, _ = hidden_states.shape

        use_cache = past_key_value is not None
        if use_cache:
            if not isinstance(past_key_value, GatedDeltaAttentionLayerCache):
                raise TypeError(
                    f'GatedDeltaAttention requires GatedDeltaAttentionLayerCache, '
                    f'got {type(past_key_value).__name__}'
                )
            conv_cache = past_key_value.conv_state if past_key_value.conv_state is not None else (None, None, None)
            S0 = past_key_value.state
        else:
            conv_cache = (None, None, None)
            S0 = None

        q, cq = self._conv(self.q_proj(hidden_states), self.q_conv, conv_cache[0])
        k, ck = self._conv(self.k_proj(hidden_states), self.k_conv, conv_cache[1])
        v, cv = self._conv(self.v_proj(hidden_states), self.v_conv, conv_cache[2])

        q = q.view(B, L, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.dv).transpose(1, 2)
        q = F.normalize(q, dim=-1) * (self.dk ** -0.5)
        k = F.normalize(k, dim=-1)

        beta = torch.sigmoid(self.b_proj(hidden_states)).transpose(1, 2)   # (B, H, L)
        g = -self.A_log.exp() * F.softplus(self.a_proj(hidden_states) + self.dt_bias)
        g = g.transpose(1, 2)                                              # (B, H, L)

        qf, kf, vf, bf, gf = (t.float() for t in (q, k, v, beta, g))
        if L == 1:
            o, S = recurrent_gated_delta_rule(qf, kf, vf, bf, gf, S0)
        elif HAS_TRITON_GDN and self.dk in (32, 64, 128) and self.dv % 64 == 0:
            o, S = chunk_gated_delta_rule_triton(qf, kf, vf, bf, gf, S0, self.chunk_size)
        else:
            o, S = chunk_gated_delta_rule(qf, kf, vf, bf, gf, S0, self.chunk_size)

        o = o.to(hidden_states.dtype).transpose(1, 2)                     # (B, L, H, Dv)
        gate = self.g_proj(hidden_states).view(B, L, self.num_heads, self.dv)
        o = self.o_norm(o, gate).reshape(B, L, -1)
        output = self.o_proj(o)

        if use_cache:
            past_key_value.update(S, (cq, ck, cv), steps=L)
            return AttentionOutput(output=output, past_key_value=past_key_value)
        # past_key_value 为 None：不建 cache，状态走 payload，由上层决定是否转 cache
        payload = {
            'state': S.detach(),
            'conv_state': tuple(c.detach() for c in (cq, ck, cv)),
        }
        return AttentionOutput(output=output, payload=payload)
