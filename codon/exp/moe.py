import torch.nn.functional as F

from codon.base import *
from codon.block.mlp import MLP
from codon.block.moe import *

import math


class ParallelExpert(nn.Module):
    '''
    A module that computes multiple expert outputs in parallel.

    This module manages weights for multiple experts and processes inputs efficiently using batch matrix multiplication.

    Attributes:
        use_gate (bool): Whether to use Gated Linear Unit (GLU) variants.
        num_experts (int): The number of experts.
        weight1 (nn.Parameter): First weight matrix with shape (num_experts, in_features, inter_dim).
        weight2 (nn.Parameter): Second weight matrix with shape (num_experts, hidden_features, out_features).
        act (nn.Module): Activation function.
        dropout (nn.Dropout): Dropout layer.
    '''

    def __init__(self, num_experts: int, in_features: int, hidden_features: int, out_features: int, use_gate: bool = False, dropout: float = 0.1) -> None:
        '''
        Initializes the ParallelExpert module.

        Args:
            num_experts (int): The number of experts.
            in_features (int): Size of each input sample.
            hidden_features (int): Size of the hidden layer.
            out_features (int): Size of each output sample.
            use_gate (bool): If True, uses SiLU activation with gating; otherwise, uses GELU.
            dropout (float): Dropout probability.
        '''
        super().__init__()
        self.use_gate = use_gate
        self.num_experts = num_experts
        self.dropout_p = dropout
        
        inter_dim = hidden_features * 2 if use_gate else hidden_features

        # [Experts, In, Hidden]
        self.weight1 = nn.Parameter(torch.empty(num_experts, in_features, inter_dim))
        self.weight2 = nn.Parameter(torch.empty(num_experts, hidden_features, out_features))
        
        self.act = nn.SiLU() if use_gate else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the experts using Kaiming Uniform initialization.
        '''
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight2[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs the forward pass for all experts in parallel.

        Args:
            x (torch.Tensor): Input tensor with shape (num_experts, capacity, in_features).

        Returns:
            torch.Tensor: Output tensor with shape (num_experts, capacity, out_features).
        '''
        # x shape: [Num_Experts, Capacity, In_Features]
        # Weight1: [Num_Experts, In_Features, Inter_Dim]
        
        # [Num_Experts, Capacity, Inter_Dim]
        h = torch.bmm(x, self.weight1)
        
        if self.use_gate:
            gate, val = h.chunk(2, dim=-1)
            h = self.act(gate) * val
        else:
            h = self.act(h)
            
        h = self.dropout(h)
        
        # Weight2: [Num_Experts, Hidden_Features, Out_Features]
        out = torch.bmm(h, self.weight2)
        
        return out

class ParallelMoE(BasicModel):
    '''
    A Parallel Mixture-of-Experts (MoE) model.

    This model routes tokens to the top-k experts and computes their outputs in parallel. It also supports shared experts and auxiliary loss for load balancing.

    Attributes:
        model_dim (int): The dimension of the model.
        top_k (int): The number of experts to route each token to.
        num_experts (int): The total number of experts.
        num_shared_experts (int): The number of shared experts that process all tokens.
        use_aux_loss (bool): Whether to use auxiliary loss for load balancing.
        capacity_factor (float): Factor to determine the capacity of each expert.
        use_gate (bool): Whether to use Gated Linear Unit (GLU) variants in experts.
        gate (nn.Linear): The gating network to route tokens to experts.
        parallel_experts (ParallelExpert): The parallel experts module.
        shared_experts (nn.ModuleList): List of shared experts, if any.
    '''

    def __init__(
        self,
        model_dim: int,
        top_k: int,
        num_experts: int,
        num_shared_experts: int = 0,
        use_aux_loss: bool = False,
        use_gate: bool = False,
        capacity_factor: float = 1.25,
    ) -> None:
        '''
        Initializes the ParallelMoE model.

        Args:
            model_dim (int): The dimension of the model.
            top_k (int): The number of experts to route each token to.
            num_experts (int): The total number of experts.
            num_shared_experts (int): The number of shared experts that process all tokens.
            use_aux_loss (bool): Whether to use auxiliary loss for load balancing.
            use_gate (bool): Whether to use Gated Linear Unit (GLU) variants in experts.
            capacity_factor (float): Factor to determine the capacity of each expert.
        '''
        super().__init__()
        self.model_dim = model_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.use_aux_loss = use_aux_loss
        self.capacity_factor = capacity_factor
        self.use_gate = use_gate

        hidden_dim = model_dim * 4

        self.gate = nn.Linear(model_dim, num_experts, bias=False)

        self.parallel_experts = ParallelExpert(
            num_experts, model_dim, hidden_dim, model_dim, use_gate=use_gate
        )

        self.shared_experts = None
        if num_shared_experts > 0:
            act_layer = "silu" if use_gate else "gelu"
            self.shared_experts = nn.ModuleList([
                MLP(
                    in_features=model_dim,
                    hidden_features=hidden_dim,
                    out_features=model_dim,
                    use_gate=use_gate,
                    act_layer=act_layer
                ) for _ in range(num_shared_experts)
            ])

    def count_params(self, trainable_only: bool = False, active_only: bool = False) -> int:
        '''
        Counts the number of parameters in the model.

        Args:
            trainable_only (bool): If True, counts only trainable parameters.
            active_only (bool): If True, counts only active parameters (parameters used during a single forward pass).

        Returns:
            int: The number of parameters.
        '''
        if not active_only:
            return super().count_params(trainable_only, active_only)
        
        total = self.gate.weight.numel()
        
        if self.shared_experts:
            total += sum(p.numel() for split in self.shared_experts for p in split.parameters())
            
        parallel_params = sum(p.numel() for p in self.parallel_experts.parameters())
        single_expert_params = parallel_params // self.num_experts
        total += single_expert_params * self.top_k
            
        return total

    @property
    def info(self) -> MoEInfo:
        '''
        Returns information about the MoE model's parameters.

        Returns:
            MoEInfo: An object containing total and active parameter counts.
        '''
        total = self.count_params(active_only=False)
        active = self.count_params(active_only=True)
        return MoEInfo(total_count=total, active_count=active)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        '''
        Performs the forward pass of the ParallelMoE model.

        This method routes tokens to experts, computes expert outputs in parallel, adds shared expert outputs, and returns the combined result.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, model_dim).

        Returns:
            MoEOutput: The output of the MoE model, containing the final output tensor and auxiliary loss.
        '''
        original_shape = x.shape
        batch, seq_len, dim = original_shape
        num_tokens = batch * seq_len
        
        x_flat = x.reshape(-1, dim)

        # Shared Experts
        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x_flat)

        # Gating
        router_logits = self.gate(x_flat) # [Tokens, Experts]
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # weights: [Tokens, TopK], indices: [Tokens, TopK]
        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Aux Loss
        aux_loss = None
        if self.use_aux_loss and self.training:
            mask = torch.zeros_like(routing_probs).scatter_(1, topk_indices, 1.0)
            density = mask.mean(dim=0)
            density_proxy = routing_probs.mean(dim=0)
            aux_loss = (self.num_experts * (density * density_proxy).sum())

        capacity = int(num_tokens * self.top_k / self.num_experts * self.capacity_factor)
        capacity = max(capacity, 4) 

        # [Tokens * TopK]
        flat_topk_indices = topk_indices.view(-1)
        
        sort_vals, sort_indices = flat_topk_indices.sort()
        
        # x_flat: [Tokens, Dim] -> [Tokens, TopK, Dim] -> [Tokens*TopK, Dim]
        x_expanded = x_flat.index_select(0, sort_indices // self.top_k)
        
        # expert_counts: [Num_Experts]
        expert_counts = torch.histc(
            flat_topk_indices.float(), 
            bins=self.num_experts, 
            min=0, 
            max=self.num_experts - 1
        ).int()

        parallel_inputs = torch.zeros(
            self.num_experts, capacity, dim, 
            dtype=x.dtype, device=x.device
        )
        
        cumsum_counts = torch.cat([torch.tensor([0], device=x.device), expert_counts.cumsum(0)])
        expert_starts = cumsum_counts[sort_vals] 
        range_indices = torch.arange(sort_vals.size(0), device=x.device)
        indices_in_expert = range_indices - expert_starts

        mask = indices_in_expert < capacity
        
        valid_indices = indices_in_expert[mask]    # [Valid_Count]
        valid_experts = sort_vals[mask]            # [Valid_Count]
        valid_inputs  = x_expanded[mask]           # [Valid_Count, Dim]

        # index: (Expert_ID, Capacity_ID)
        parallel_inputs[valid_experts, valid_indices] = valid_inputs

        parallel_outputs = self.parallel_experts(parallel_inputs)
        # [Num_Experts, Capacity, Dim]

        # [Tokens * TopK, Dim]
        combined_output = torch.zeros(
            num_tokens * self.top_k, dim, 
            dtype=x.dtype, device=x.device
        )
        
        # parallel_outputs[valid_experts, valid_indices]
        valid_outputs = parallel_outputs[valid_experts, valid_indices]
        
        original_positions = sort_indices[mask] # [Valid_Count]

        token_ids = original_positions.div(self.top_k, rounding_mode='floor')
        
        # [Tokens * TopK]
        flat_weights = topk_weights.view(-1)
        valid_weights = flat_weights[original_positions].unsqueeze(-1) # W
        
        weighted_output = valid_outputs * valid_weights
        
        final_output = torch.zeros_like(x_flat)
        final_output.index_add_(0, token_ids, weighted_output)

        final_output = final_output + shared_output

        # [Batch, Seq, Dim]
        return MoEOutput(
            output=final_output.reshape(original_shape), 
            aux_loss=aux_loss
        )
