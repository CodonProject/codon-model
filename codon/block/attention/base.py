from codon import *
from codon.ops import AttentionOutput
from codon.block.embedding import BasicEmbedding
from codon.model.cache import BasicLayerCache, LinearAttentionLayerCache

from abc import ABC, abstractmethod


class BasicAttention(BasicModel, ABC):
    '''
    注意力模块的抽象契约基类。

    统一了所有注意力机制共用的几何约定（head 划分、KV 压缩维度等）与
    forward 签名，并约定每条路径都应返回 codon.ops.AttentionOutput。
    具体机制（softmax / linear kernel / Fourier ...）由子类各自实现。

    继承者应：
        - 在 __init__ 中调用 super().__init__() 完成几何推导；
        - 实现 abstract forward；
        - 覆写 cache_type() 返回本机制对应的 BasicLayerCache 子类。

    Attributes:
        hidden_size (int): 隐藏维度。
        num_heads (int): 注意力头数。
        num_kv_heads (int): GQA KV 头数（默认等于 num_heads）。
        head_dim (int): 每头维度 = hidden_size // num_heads。
        kv_dim (int): KV 投影输出维度 = num_kv_heads * head_dim。
        num_kv_queries (int): 每组 KV 复用的查询头数 = num_heads // num_kv_heads。
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        assert num_heads % num_kv_heads == 0, 'num_heads must be divisible by num_kv_heads'

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

    # ---- 供推理链路选择缓存 ----
    @classmethod
    def cache_type(cls) -> type:
        '''返回本机制应使用的 BasicLayerCache 子类（供 build_cache 路由）。'''
        return BasicLayerCache

    # ---- 几何工具 ----
    def reshape_q(
        self,
        hidden_states: torch.Tensor,
        q_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        '''[B, L, D] -> [B, num_heads, L, head_dim]'''
        q = hidden_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        return q.transpose(1, 2)

    def reshape_kv(
        self,
        kv_states: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        '''[B, L, D] -> [B, num_kv_heads, L, head_dim]'''
        kv = kv_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return kv.transpose(1, 2)

    @abstractmethod
    def forward(self, *args, **kwargs) -> AttentionOutput:
        '''执行本机制的注意力前向，返回 AttentionOutput。'''
        raise NotImplementedError


class BasicLinearAttention(BasicAttention):
    '''
    Linear Attention（Katharopoulos et al., 2020）的独立实现。

    用可分解核把注意力从 O(L²) 降到 O(L)：
        标准 softmax 注意力  y_i = sum_j softmax(q_i · k_j) v_j
        线性核近似           y_i = φ(q_i)ᵀ S_i / (φ(q_i)ᵀ z_i)
        其中  S_i = sum_{j<=i} φ(k_j) v_jᵀ   （causal 前缀累加状态）
              z_i = sum_{j<=i} φ(k_j)        （归一化分母）

    默认核为 φ(x) = elu(x) + 1。本模块为独立实现，不依赖 apply_attention /
    softmax 路径；KV cache 使用 LinearAttentionLayerCache（固定大小状态递推），
    因而 decode 复杂度与序列长度无关。

    Attributes:
        q_proj (nn.Linear): Query 投影。
        k_proj (nn.Linear): Key 投影（GQA 压缩到 num_kv_heads）。
        v_proj (nn.Linear): Value 投影（GQA 压缩到 num_kv_heads）。
        o_proj (nn.Linear): 输出投影。
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        feature_map: str = 'elu',
        eps: float = 1e-6,
        **kwargs,
    ):
        '''
        Args:
            hidden_size (int): 隐藏维度。
            num_heads (int): 注意力头数。
            num_kv_heads (int, optional): KV 头数；None 表示与 num_heads 相同。
            dropout (float): 输出 dropout 概率。
            bias (bool): 投影层是否带 bias。
            feature_map (str): 核特征映射，支持 'elu'（elu(x)+1）或 'identity'。
            eps (float): 归一化分母防零项。
        '''
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )

        self.dropout = nn.Dropout(dropout)
        self.feature_map = feature_map.lower()
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    @classmethod
    def cache_type(cls) -> type:
        return LinearAttentionLayerCache

    # ---- 核特征映射 ----
    def _feature(self, x: torch.Tensor) -> torch.Tensor:
        '''φ(x)：把 [B, H, L, D] 映射到特征空间（D 不变）。'''
        if self.feature_map == 'elu':
            return F.elu(x, alpha=1.0) + 1.0
        if self.feature_map == 'identity':
            return x
        raise ValueError(f"Unsupported feature_map: {self.feature_map!r}. Use 'elu' or 'identity'.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: BasicEmbedding = None,
        embedding_start: int = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: Optional[BasicLayerCache] = None,
    ) -> AttentionOutput:
        '''
        线性注意力前向。

        支持两种调用形态：
          - Prefill / 全序列：past_key_value 为 None（或空 state），对整段做
            前缀状态递推，得到逐位置输出；
          - Decode 单步：传入已含状态的 LinearAttentionLayerCache，只推进一个
            token，返回该 token 的注意力输出并更新 cache。

        注意：本实现不消费 attention_mask / position_emb——线性核没有 softmax
        掩码语义，RoPE 也不与状态递推线性核直接组合。
        '''
        batch_size, q_len, _ = hidden_states.shape

        # 1. 投影 + reshape 到 [B, H, L, D]
        q = self.reshape_q(self.q_proj(hidden_states), q_len, batch_size)   # query 头数 H
        k = self.reshape_kv(self.k_proj(hidden_states), q_len, batch_size)  # KV 头数 Hk
        v = self.reshape_kv(self.v_proj(hidden_states), q_len, batch_size)

        q_feat = self._feature(q)
        k_feat = self._feature(k)

        # 是否启用归一化分母（elu+1 核需要；identity 核数值无界，禁用归一化反而常见）
        use_norm = self.feature_map == 'elu'

        # 2. cache 准备
        if past_key_value is None:
            past_key_value = LinearAttentionLayerCache()
        if not isinstance(past_key_value, LinearAttentionLayerCache):
            raise TypeError(
                f'BasicLinearAttention requires LinearAttentionLayerCache, got {type(past_key_value).__name__}'
            )

        # 3. 逐 token causal 状态递推
        outputs = []
        for t in range(q_len):
            q_t = q_feat[:, :, t:t+1, :]     # [B, H, 1, D]
            k_t = k_feat[:, :, t:t+1, :]     # [B, Hk, 1, D]
            v_t = v[:, :, t:t+1, :]          # [B, Hk, 1, D]

            # GQA：把每个 KV 头平铺 num_kv_queries 份，得到与查询头数一致的头序
            # （顺序与 MHA 的 [kv0×G, kv1×G, ...] reshape 展开一致）
            if self.num_kv_queries > 1:
                k_g = k_t.repeat_interleave(self.num_kv_queries, dim=1)
                v_g = v_t.repeat_interleave(self.num_kv_queries, dim=1)
            else:
                k_g, v_g = k_t, v_t

            state_inc = torch.matmul(k_g.transpose(-1, -2), v_g)   # [B, H, D, Dv]
            norm_inc = k_g.sum(dim=-1) if use_norm else None       # [B, H, 1]

            # 用「含本步的状态」算输出 → 实现 causal（当前 token 可见自身）
            S_t = state_inc if past_key_value.state is None else (past_key_value.state + state_inc)
            z_t = norm_inc if norm_inc is None or past_key_value.norm_state is None else (past_key_value.norm_state + norm_inc)

            num = torch.matmul(q_t, S_t)                           # [B, H, 1, Dv]
            if z_t is not None:
                denom = (q_t * z_t.unsqueeze(-1)).sum(dim=-1, keepdim=True)  # [B, H, 1, 1]
                num = num / (denom + self.eps)

            out = num.transpose(1, 2).reshape(batch_size, 1, self.num_heads * self.head_dim)  # [B,1,D]
            outputs.append(out)

            # 累积状态入 cache（供下一 token 使用）
            past_key_value.update(
                state_inc.detach(),
                steps=1,
                norm_increment=norm_inc.detach() if norm_inc is not None else None,
            )

        output = torch.cat(outputs, dim=1)                         # [B, L, D]
        output = self.o_proj(self.dropout(output))

        return AttentionOutput(
            output=output,
            attention_weights=None,
            past_key_value=past_key_value,
        )
