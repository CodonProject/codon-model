from codon import *

from codon.utils import safecode
from codon.block import (
    BasicEmbedding,
    MultiHeadAttention, AttentionOutput,
    MultiHeadFourier,
    MLP,
    MoE, MoEOutput
)

from typing import Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowOutput:
    '''
    Output from the flow (feed-forward) layer.

    Args:
        output (torch.Tensor): The output tensor.
        aux_loss (Optional[torch.Tensor], optional): Auxiliary loss (e.g., for MoE). Defaults to None.
    '''
    output: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None

@dataclass
class TransformerDecoderOutput:
    '''
    Output from the Transformer Decoder layer.

    Args:
        idx (str): Identifier for the layer.
        output (torch.Tensor): The output hidden states.
        attention_weights (Optional[torch.Tensor], optional): Attention weights. Defaults to None.
        attention_mask (Optional[torch.Tensor], optional): Attention mask used. Defaults to None.
        aux_loss (Optional[torch.Tensor], optional): Auxiliary loss from the flow layer. Defaults to None.
        past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): KV cache for the current step. Defaults to None.
        use_emb (Optional[BasicEmbedding], optional): Positional embedding module used. Defaults to None.
        emb_start (Optional[int], optional): Start position for embedding. Defaults to 0.
        emb_pos (Optional[torch.Tensor], optional): Explicit positions for embedding. Defaults to None.
    '''
    idx: str
    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    use_emb: Optional[BasicEmbedding] = None
    emb_start: Optional[int] = 0
    emb_pos: Optional[torch.Tensor] = None


class _TransformerDecoder(BasicModel):
    '''
    Base class for Transformer Decoder layers.

    Implements the standard decoder block with self-attention and a configurable feed-forward (flow) network.

    Attributes:
        idx (str): Layer identifier.
        model_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads for GQA.
        use_qk_norm (bool): Whether to use QK normalization.
        use_attn_gate (bool): Whether to use attention gating.
        attn_norm (nn.RMSNorm): Pre-attention normalization.
        attn (MultiHeadAttention): Multi-head attention module.
        fn_norm (nn.RMSNorm): Pre-feed-forward normalization.
        dropout (nn.Dropout): Dropout layer.
    '''
    def __init__(
        self,
        model_dim: int = 1024, 
        num_heads: int = 16, 
        num_kv_heads: int = 4,
        use_qk_norm=True,
        use_attn_gate=False,
        dropout=0.1,
        attn_type: str = 'multihead',
        attn_bias: bool=False,
        idx: Union[int, str] = None
    ):
        '''
        Initializes the Transformer Decoder layer.

        Args:
            model_dim (int, optional): Model dimension. Defaults to 1024.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            num_kv_heads (int, optional): Number of KV heads for GQA. Defaults to 4.
            use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Defaults to True.
            use_attn_gate (bool, optional): Whether to apply gating to attention output. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            idx (Union[int, str], optional): Layer identifier. Defaults to None.
        '''
        super().__init__()

        self.idx = str(idx) if idx is not None else safecode()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_attn_gate = use_attn_gate
        self.attn_bias = attn_bias
        self.attn_type = attn_type.lower()

        self.attn_norm = nn.RMSNorm(model_dim)
        
        if self.attn_type in ['multihead', 'mha']:
            self.attn = MultiHeadAttention(
                hidden_size=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_qk_norm=use_qk_norm,
                use_gate=use_attn_gate,
                dropout=dropout,
                is_causal=True,
                bias=attn_bias
            )

        elif self.attn_type in ['fourier', 'mhf']:
            self.attn = MultiHeadFourier(
                hidden_size=model_dim,
                num_heads=num_heads
            )

        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}. Choose from 'multihead', 'linear', or 'gated_linear'.")

        self.fn_norm = nn.RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: BasicEmbedding = None,
        embedding_start: Union[int, torch.Tensor] = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> TransformerDecoderOutput:
        '''
        Forward pass of the Transformer Decoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            position_emb (BasicEmbedding, optional): Positional embedding module. Defaults to None.
            embedding_start (int, optional): Starting position for embedding. Defaults to 0.
            embedding_pos (torch.Tensor, optional): Explicit positions for embedding. Defaults to None.
            past_key_value (tuple[torch.Tensor, torch.Tensor], optional): Past KV cache. Defaults to None.
            use_cache (bool, optional): Whether to use/update KV cache. Defaults to False.

        Returns:
            TransformerDecoderOutput: Output object containing hidden states and other information.
        '''
        x = self.attn_norm(hidden_states)

        attention_output: AttentionOutput = self.attn(
            hidden_states=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_emb=position_emb,
            embedding_start=embedding_start,
            embedding_pos=embedding_pos,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output.output)

        x = self.fn_norm(hidden_states)
        flow_output = self.flow(x)
        hidden_states = hidden_states + flow_output.output

        return TransformerDecoderOutput(
            idx=self.idx,
            output=hidden_states,
            attention_weights=attention_output.attention_weights,
            attention_mask=attention_mask,
            aux_loss=flow_output.aux_loss,
            past_key_value=attention_output.past_key_value,
            use_emb=position_emb,
            emb_start=embedding_start,
            emb_pos=embedding_pos
        )
    
    def forward_dc(
        self, 
        data: TransformerDecoderOutput, 
        current_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> TransformerDecoderOutput:
        '''
        Forward pass utilizing a TransformerDecoderOutput object for chained execution.

        Args:
            data (TransformerDecoderOutput): Input data from the previous layer.
            current_layer_kv (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): KV cache for the current layer. Defaults to None.

        Returns:
            TransformerDecoderOutput: Output object for the next layer.
        '''
        use_cache = current_layer_kv is not None
        need_attn_weight = data.attention_weights is not None
        return self.forward(
            hidden_states=data.output,
            attention_mask=data.attention_mask,
            output_attentions=need_attn_weight,
            position_emb=data.use_emb,
            embedding_start=data.emb_start,
            embedding_pos=data.emb_pos,
            past_key_value=current_layer_kv,
            use_cache=use_cache
        )

    def flow(self, x: torch.Tensor) -> FlowOutput:
        '''
        Abstract method for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            FlowOutput: Output from the flow layer.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        '''
        raise NotImplementedError()


class TransformerDenseDecoder(_TransformerDecoder):
    '''
    Transformer Decoder layer with a dense MLP feed-forward network.

    Attributes:
        mlp (MLP): Multi-layer perceptron module.
    '''
    def __init__(
        self, 
        model_dim: int = 1024, 
        num_heads: int = 16, 
        num_kv_heads: int = 4,
        mlp_ratio: float = 4.0, 
        use_mlp_gate: bool = False,
        use_qk_norm: bool = True,
        use_attn_gate: bool = False,
        attn_type: str = 'multihead',
        use_swiglu: bool = False,
        dropout: float = 0.1,
        attn_bias: bool=False,
        mlp_bias: bool=False,
        idx: Union[int, str] = None
    ):
        '''
        Initializes the TransformerDenseDecoder.

        Args:
            model_dim (int, optional): Model dimension. Defaults to 1024.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            num_kv_heads (int, optional): Number of KV heads for GQA. Defaults to 4.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to model dimension. Defaults to 4.0.
            use_mlp_gate (bool, optional): Whether to use gating in MLP. Defaults to False.
            use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Defaults to True.
            use_attn_gate (bool, optional): Whether to apply gating to attention output. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            idx (Union[int, str], optional): Layer identifier. Defaults to None.
        '''
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=use_qk_norm,
            use_attn_gate=use_attn_gate,
            dropout=dropout,
            attn_type=attn_type,
            idx=idx,
            attn_bias=attn_bias
        )
        self.mlp = MLP(
            in_features=model_dim,
            hidden_features=int(model_dim * mlp_ratio),
            out_features=model_dim,
            use_gate=use_mlp_gate,
            dropout=dropout,
            bias=mlp_bias
        ) if not use_swiglu else MLP.SwiGLU(model_dim)
    
    def flow(self, x: torch.Tensor) -> FlowOutput:
        '''
        Forward pass of the dense MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            FlowOutput: Output from the MLP (aux_loss is None).
        '''
        return FlowOutput(output=self.mlp(x), aux_loss=None)


class TransformerMoEDecoder(_TransformerDecoder):
    '''
    Transformer Decoder layer with a Mixture-of-Experts (MoE) feed-forward network.

    Attributes:
        moe (MoE): Mixture-of-Experts module.
    '''
    def __init__(
        self, 
        model_dim: int = 1024, 
        num_heads: int = 16, 
        num_kv_heads: int = 4,
        top_k: int = 2,
        num_experts: int = 8,
        num_shared_experts: int = 0,
        use_aux_loss: bool = False,
        use_expert_gate: bool = False,
        use_qk_norm: bool = True,
        use_attn_gate: bool = False,
        attn_type: str = 'multihead',
        dropout: float = 0.1,
        attn_bias: bool=False,
        idx: Union[int, str] = None
    ):
        '''
        Initializes the TransformerMoEDecoder.

        Args:
            model_dim (int, optional): Model dimension. Defaults to 1024.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            num_kv_heads (int, optional): Number of KV heads for GQA. Defaults to 4.
            top_k (int, optional): Number of experts to select per token. Defaults to 2.
            num_experts (int, optional): Total number of experts. Defaults to 8.
            num_shared_experts (int, optional): Number of shared experts. Defaults to 0.
            use_aux_loss (bool, optional): Whether to compute auxiliary loss for load balancing. Defaults to False.
            use_expert_gate (bool, optional): Whether to use Gated MLP for experts. Defaults to False.
            use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Defaults to True.
            use_attn_gate (bool, optional): Whether to apply gating to attention output. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            idx (Union[int, str], optional): Layer identifier. Defaults to None.
        '''
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=use_qk_norm,
            use_attn_gate=use_attn_gate,
            dropout=dropout,
            attn_type=attn_type,
            idx=idx,
            attn_bias=attn_bias
        )
        self.moe = MoE(
            model_dim=model_dim,
            top_k=top_k,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            use_aux_loss=use_aux_loss,
            use_gate=use_expert_gate
        )
    
    def flow(self, x: torch.Tensor) -> FlowOutput:
        '''
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            FlowOutput: Output from the MoE, including auxiliary loss if enabled.
        '''
        output: MoEOutput = self.moe(x)
        return FlowOutput(output=output.output, aux_loss=output.aux_loss)
