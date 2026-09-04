from codon import *
from codon.block.transformer import TransformerDenseDecoder
from codon.block.embedding   import RotaryEmbedding
from codon.utils.tokens      import PackedTokenizer
from codon.model.cache       import ModelCache, build_cache
from codon.model.types.language import CausalLanguageModel, CausalLanguageModelOutput


class MotifA1Tokenizer(PackedTokenizer):
    
    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['motif.vocab'],
        'repo_type': 'model'
    }


class MotifA1(CausalLanguageModel):

    __remote_resource__ = {
        'repo': 'CodonProject/MotifA1-SFT',
        'files': ['MotifA1_SFT.safetensors'],
        'repo_type': 'model'
    }

    def __init__(
        self,
        vocab_size: int = 2**13,
        model_dim: int = 768,
        num_layers: int = 16,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.position_emb = RotaryEmbedding(model_dim // num_heads)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList([
            TransformerDenseDecoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_qk_norm=True,
                use_attn_gate=False,
                use_swiglu=True,
                dropout=dropout,
                attn_bias=False,
                attn_type='mha',
                idx=str(idx)
            )
            for idx in range(num_layers)
        ])

        self.norm = nn.RMSNorm(model_dim)
        self.proj_out = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.proj_out.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])

    # 早期版 MotifA1 权重布局与本类现行结构存在三点无损差异，加载前就地重排：
    #   1) MLP：早期为分离 gate_proj/up_proj（本类 use_swiglu=True 用融合 gate_up_proj）。
    #      gate_up_proj.weight = cat([up_proj.weight, gate_proj.weight], dim=0)
    #      （codon SwiGLU 前向 out,gate=split(x)，前半不激活 = up，后半 silu = gate，故 up 在前）
    #   2) decoder 层 RMSNorm：早期参数名 weight（codon.block.norm.RMSNorm 现为 gamma）。
    #      顶层 norm 与 attention 内 q/k norm 为 torch.nn.RMSNorm，参数本就叫 weight，无需改名。
    #   3) token_emb.weight：早期 tie 时只落盘 proj_out.weight，由它补一份。
    #  注意：本方法不改 codon 库任何模块结构，仅做 checkpoint 键名/形状适配。
    def load(self, path, strict=False):
        if isinstance(path, str) and path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                raw = load_file(path)
            except Exception as e:
                print(f'[MotifA1] 读取 {path} 失败: {e}；回落默认加载')
                return super().load(path, strict=strict)

            mapped = self._adapt_legacy_state_dict(raw)
            if mapped is not None:
                missing = [k for k in self.state_dict() if k not in mapped]
                if missing:
                    msg = '无法从 checkpoint 解析到权重: ' + ', '.join(missing)
                    if strict:
                        raise RuntimeError(f'[MotifA1] {msg}')
                    print(f'[MotifA1] 警告: {msg}')
                self.load_state_dict(mapped, strict=strict)
                print(f'[MotifA1] 已从旧布局 checkpoint 重排加载: {path}')
                return self
        return super().load(path, strict=strict)

    def _adapt_legacy_state_dict(self, raw):
        '''把旧式（gate/up 分离 + decoder norm 用 weight 命名）checkpoint 无损映射到当前结构。
        若文件已匹配当前结构（融合 gate_up_proj）则返回 None，交由默认加载。'''
        expected = self.state_dict()
        exp_keys = set(expected)

        need_split_merge = (
            any(k.endswith('.mlp.gate_up_proj.weight') for k in exp_keys)
            and any(k.endswith('.mlp.up_proj.weight') or k.endswith('.mlp.gate_proj.weight') for k in raw)
            and not any(k.endswith('.mlp.gate_up_proj.weight') for k in raw)
        )
        if not need_split_merge:
            return None

        def source_for(model_key):
            if model_key in raw:
                return raw[model_key]
            # tie 时 token_emb 只落盘一份 proj_out
            if model_key == 'token_emb.weight' and 'proj_out.weight' in raw:
                return raw['proj_out.weight']
            # decoder 层 RMSNorm: .gamma <- 旧式 .weight（顶层/attention norm 为 torch nn.RMSNorm，key 即 weight，已同名命中）
            if model_key.endswith('.attn_norm.gamma') or model_key.endswith('.fn_norm.gamma'):
                alt = model_key.rsplit('.', 1)[0] + '.weight'
                if alt in raw:
                    return raw[alt]
            # 融合 MLP: gate_up_proj <- up_proj + gate_proj（up 在前，与 SwiGLU split 语义一致）
            suf = '.mlp.gate_up_proj.weight'
            if model_key.endswith(suf):
                pre = model_key[:-len(suf)]
                up = raw.get(pre + '.mlp.up_proj.weight')
                ga = raw.get(pre + '.mlp.gate_proj.weight')
                if up is not None and ga is not None:
                    return torch.cat([up, ga], dim=0)
            return None

        mapped = {}
        for k in exp_keys:
            v = source_for(k)
            if v is not None:
                mapped[k] = v
        return mapped

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: Union[int, torch.Tensor] = 0,
        past_key_values: Optional[ModelCache] = None,
        output_attentions: bool = False
    ) -> CausalLanguageModelOutput:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        all_attentions = [] if output_attentions else None
        aux_loss = None

        for i, layer in enumerate(self.decoder):
            layer_past = None
            if isinstance(past_key_values, ModelCache):
                if past_key_values[i] is None:
                    past_key_values[i] = build_cache(layer.attn)
                layer_past = past_key_values[i]
            
            out = layer(
                hidden_states=x,
                attention_mask=mask,
                output_attentions=output_attentions,
                position_emb=self.position_emb,
                embedding_start=start_pos,
                past_key_value=layer_past,
            )
            
            x = out.output
            
            if output_attentions:
                all_attentions.append(out.attention_weights)
            
            if out.aux_loss is not None:
                if aux_loss is None:
                    aux_loss = out.aux_loss
                else:
                    aux_loss += out.aux_loss

        x = self.norm(x)
        logits = self.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            aux_loss=aux_loss,
            attentions=all_attentions
        )