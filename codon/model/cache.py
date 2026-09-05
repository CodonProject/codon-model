from codon import *


class BasicLayerCache:
    @property
    def seq_length(self) -> int:
        raise NotImplementedError
    
    def update(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'BasicLayerCache':
        raise NotImplementedError


class KVLayerCache(BasicLayerCache):
    '''标准 MHA/GQA 的 KV 缓存 (K, V 独立存储)'''
    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
    
    @property
    def seq_length(self) -> int:
        return self.k.shape[2] if self.k is not None else 0  # [Batch, Head, Seq, Dim]
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, dim: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = k_new.detach()
            self.v = v_new.detach()
        else:
            if self.k.device != k_new.device:
                self.to(k_new.device)
            self.k = torch.cat([self.k, k_new], dim=dim)
            self.v = torch.cat([self.v, v_new], dim=dim)
        return self.k, self.v
    
    def reset(self):
        self.k = None
        self.v = None
        
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'KVLayerCache':
        if self.k is not None:
            self.k = self.k.to(device=device, dtype=dtype)
            self.v = self.v.to(device=device, dtype=dtype)
        return self


class TensorLayerCache(BasicLayerCache):
    '''单张量缓存。适用于 MLA (kv_latent) 和优化后的 KEV (单个 KV 矩阵)'''
    def __init__(self, concat_dim: int = 1):
        self.tensor: Optional[torch.Tensor] = None
        self.concat_dim = concat_dim
    
    @property
    def seq_length(self) -> int:
        if self.tensor is None:
            return 0
        return self.tensor.shape[self.concat_dim]
    
    def update(self, tensor_new: torch.Tensor) -> torch.Tensor:
        if self.tensor is None:
            self.tensor = tensor_new.detach()
        else:
            # 自动检测设备不一致并进行转移
            if self.tensor.device != tensor_new.device:
                self.to(tensor_new.device)
            self.tensor = torch.cat([self.tensor, tensor_new], dim=self.concat_dim)
        return self.tensor
    
    def reset(self):
        self.tensor = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'TensorLayerCache':
        if self.tensor is not None:
            self.tensor = self.tensor.to(device=device, dtype=dtype)
        return self


class FourierLayerCache(BasicLayerCache):
    '''Fourier Mixing 特有的缓存 (卷积状态 + V缓存 + G缓存)'''
    def __init__(self):
        self.conv_state: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.g_cache: Optional[torch.Tensor] = None

    @property
    def seq_length(self) -> int:
        return self.v_cache.shape[2] if self.v_cache is not None else 0

    def update(self, new_conv_state: torch.Tensor, v_new: torch.Tensor, g_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.conv_state = new_conv_state.detach()
        if self.v_cache is None:
            self.v_cache = v_new.detach()
            self.g_cache = g_new.detach()
        else:
            # 自动检测设备不一致并进行转移
            if self.v_cache.device != v_new.device:
                self.to(v_new.device)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
            self.g_cache = torch.cat([self.g_cache, g_new], dim=2)
        return self.conv_state, self.v_cache, self.g_cache

    def reset(self):
        self.conv_state = None
        self.v_cache = None
        self.g_cache = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'FourierLayerCache':
        if self.conv_state is not None:
            self.conv_state = self.conv_state.to(device=device, dtype=dtype)
        if self.v_cache is not None:
            self.v_cache = self.v_cache.to(device=device, dtype=dtype)
            self.g_cache = self.g_cache.to(device=device, dtype=dtype)
        return self


class LinearAttentionLayerCache(BasicLayerCache):
    '''
    线性注意力 (Linear Attention / Mamba / Recurrent) 缓存。

    线性注意力的 Cache 是固定大小的状态矩阵，而非变长的 K/V concat：
        numerator  S_t = S_{t-1} + K_t^T * V_t          [Batch, Head, D_k, D_v]
        normalize  z_t = z_{t-1} + sum(K_t, dim=-1)     [Batch, Head, D_k]  （可选，用于 elu+1 核的归一化）
    '''
    def __init__(self):
        self.state: Optional[torch.Tensor] = None    # numerator [B, H, Dk, Dv]
        self.norm_state: Optional[torch.Tensor] = None  # normalize z [B, H, Dk]
        self._seq_len = 0

    @property
    def seq_length(self) -> int:
        return self._seq_len

    def update(
        self,
        state_increment: torch.Tensor,
        steps: int = 1,
        norm_increment: torch.Tensor = None,
    ) -> torch.Tensor:
        '''
        累加线性注意力状态。

        Args:
            state_increment: numerator 增量 [B, H, Dk, Dv]。
            steps: 本步推进的 token 数（默认 1，供 seq_length 计数）。
            norm_increment: 归一化 z 增量 [B, H, Dk]；None 表示不使用归一化分母。
        '''
        if self.state is None:
            self.state = state_increment.detach()
        else:
            if self.state.device != state_increment.device:
                self.to(state_increment.device)
            self.state = self.state + state_increment

        if norm_increment is not None:
            if self.norm_state is None:
                self.norm_state = norm_increment.detach()
            else:
                if self.norm_state.device != norm_increment.device:
                    self.norm_state = self.norm_state.to(norm_increment.device)
                self.norm_state = self.norm_state + norm_increment

        self._seq_len += steps
        return self.state

    def reset(self):
        self.state = None
        self.norm_state = None
        self._seq_len = 0

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'LinearAttentionLayerCache':
        if self.state is not None:
            self.state = self.state.to(device=device, dtype=dtype)
        if self.norm_state is not None:
            self.norm_state = self.norm_state.to(device=device, dtype=dtype)
        return self


class HCALayerCache(BasicLayerCache):
    def __init__(self, fp4_storage: bool = True):
        self.fp4_storage = fp4_storage
        self.quantized: Optional[torch.Tensor] = None
        self.min_val: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.raw_tensor: Optional[torch.Tensor] = None
        self._seq_blocks = 0

    @property
    def seq_length(self) -> int:
        return self._seq_blocks

    def update_raw(self, tensor_new: torch.Tensor) -> torch.Tensor:
        if self.raw_tensor is None:
            self.raw_tensor = tensor_new.detach()
        else:
            if self.raw_tensor.device != tensor_new.device:
                self.to(tensor_new.device)
            self.raw_tensor = torch.cat([self.raw_tensor, tensor_new], dim=1)
        self._seq_blocks = self.raw_tensor.shape[1]
        return self.raw_tensor

    def update_fp4(self, quantized: torch.Tensor, min_val: torch.Tensor, scale: torch.Tensor, num_blocks: int):
        if self.quantized is None:
            self.quantized = quantized.detach()
            self.min_val = min_val.detach()
            self.scale = scale.detach()
        else:
            if self.quantized.device != quantized.device:
                self.to(quantized.device)
            # quantized: [B, num_blocks, D]
            # min_val & scale: [B, num_blocks, 1]
            self.quantized = torch.cat([self.quantized, quantized], dim=1)
            self.min_val = torch.cat([self.min_val, min_val], dim=1)
            self.scale = torch.cat([self.scale, scale], dim=1)
        self._seq_blocks = self.quantized.shape[1]
        return self.quantized, self.min_val, self.scale

    def reset(self):
        self.quantized = None
        self.min_val = None
        self.scale = None
        self.raw_tensor = None
        self._seq_blocks = 0

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'HCALayerCache':
        if self.quantized is not None:
            self.quantized = self.quantized.to(device=device)
            self.min_val = self.min_val.to(device=device, dtype=dtype)
            self.scale = self.scale.to(device=device, dtype=dtype)
        if self.raw_tensor is not None:
            self.raw_tensor = self.raw_tensor.to(device=device, dtype=dtype)
        return self

    
class CSALayerCache(BasicLayerCache):
    def __init__(self):
        self.kv_blocks: Optional[torch.Tensor] = None

    @property
    def seq_length(self) -> int:
        return self.kv_blocks.shape[1] if self.kv_blocks is not None else 0
    
    def update(self, new_kv_blocks: torch.Tensor) -> torch.Tensor:
        if self.kv_blocks is None:
            self.kv_blocks = new_kv_blocks.detach()
        else:
            if self.kv_blocks.device != new_kv_blocks.device:
                self.to(new_kv_blocks.device)
            self.kv_blocks = torch.cat([self.kv_blocks, new_kv_blocks], dim=1)
        return self.kv_blocks
    
    def reset(self):
        self.kv_blocks = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'CSALayerCache':
        if self.kv_blocks is not None:
            self.kv_blocks = self.kv_blocks.to(device=device, dtype=dtype)
        return self


class ModelCache:
    def __init__(self):
        self.layer_caches: Dict[int, BasicLayerCache] = {}
        self.device: Optional[torch.device] = None
    
    @property
    def seq_length(self) -> int:
        if len(self.layer_caches) > 0:
            first_cache = next(iter(self.layer_caches.values()))
            return first_cache.seq_length
        return 0

    def __len__(self) -> int:
        return len(self.layer_caches)
    
    def __getitem__(self, layer_idx: int) -> Optional[BasicLayerCache]:
        return self.layer_caches.get(layer_idx, None)
    
    def __setitem__(self, layer_idx: int, cache: BasicLayerCache):
        if self.device is not None:
            cache.to(self.device)
        self.layer_caches[layer_idx] = cache
    
    def reset(self):
        for c in self.layer_caches.values(): 
            c.reset()

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'ModelCache':
        self.device = device
        for k, v in self.layer_caches.items():
            self.layer_caches[k] = v.to(device, dtype)
        return self


def build_cache(layer: object) -> BasicLayerCache:
    '''
    按 layer 的机制特征创建对应 KV 缓存。

    - MultiHeadFourier → FourierLayerCache
    - MultiHeadAttentionKEV → TensorLayerCache(concat_dim=2)
    - MLA（kv_lora_rank > 0）→ TensorLayerCache(concat_dim=1)
    - HCA（use_hca）→ HCALayerCache
    - CSA（use_csa）→ CSALayerCache
    - 标准 MHA / GQA（含纯 MultiHeadAttention 与 legacy 各模式）→ KVLayerCache
    - BasicLinearAttention → LinearAttentionLayerCache

    注意：为避免与 codon.block.attention 的模块级 import 环，本函数
    （而非模块顶层）延迟 import 具体注意力类型。
    '''
    if hasattr(layer, 'module'):
        layer = layer.module

    # 1) 属性路由（先）：MLA/HCA/CSA 特征可直接由属性判定
    if getattr(layer, 'use_hca', False):
        return HCALayerCache(fp4_storage=getattr(layer, 'hca_fp4_storage', True))
    if getattr(layer, 'use_csa', False):
        return CSALayerCache()
    if getattr(layer, 'kv_lora_rank', 0) > 0:
        return TensorLayerCache(concat_dim=1)

    # 2) 声明式路由：若 layer 覆写了 cache_type() 类方法（非契约默认值），优先采用。
    #    供 BasicAttention 系各机制声明自己的缓存类型（如线性注意力 → LinearAttentionLayerCache）。
    declared = getattr(type(layer), 'cache_type', None)
    if declared is not None:
        from codon.block.attention.base import BasicAttention as _BasicAttn
        declared_type = declared()
        if issubclass(type(layer), _BasicAttn) and declared_type is not BasicLayerCache:
            if declared_type is LinearAttentionLayerCache:
                return LinearAttentionLayerCache()
            return declared_type()

    # 3) 类型路由（后）：标准 MHA/GQA / KEV / Fourier
    from codon.block.attention.mha import MultiHeadAttention as _MHA
    from codon.block.attention._legacy import MultiHeadAttentionLegacy as _Legacy
    from codon.block.attention._legacy import MultiHeadAttentionKEV as _KEV
    from codon.block.fourier import MultiHeadFourier as _MHF

    if isinstance(layer, _KEV):
        return TensorLayerCache(concat_dim=2)
    if isinstance(layer, _MHF):
        return FourierLayerCache()
    if isinstance(layer, _MHA):
        return KVLayerCache()
    if isinstance(layer, _Legacy):
        # legacy 多机制实例：无 hca/csa/lora 特征时即标准 MHA/GQA 分支
        return KVLayerCache()

    raise TypeError(f'Unsupported layer type for cache creation: {type(layer).__name__}')