import torch.nn.functional as F

from codon.base import *

from dataclasses import dataclass
from typing      import Union

import math


@dataclass
class MoEOutput:
    output: torch.Tensor
    aux_loss: Union[torch.Tensor, None]

@dataclass
class MoEInfo:
    total_count: int
    active_count: int


class ParallelExpert(nn.Module):
    def __init__(self, num_experts, in_features, hidden_features, out_features, use_gate=False, dropout=0.1):
        super().__init__()
        self.use_gate = use_gate
        self.num_experts = num_experts
        
        # 如果使用 SwiGLU (use_gate=True)，中间维度需要翻倍
        inter_dim = hidden_features * 2 if use_gate else hidden_features

        # 权重形状: [Experts, In, Hidden] -> 这允许我们使用 torch.bmm
        self.weight1 = nn.Parameter(torch.empty(num_experts, in_features, inter_dim))
        self.weight2 = nn.Parameter(torch.empty(num_experts, hidden_features, out_features))
        
        self.act = nn.SiLU() if use_gate else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化技巧：对并行权重进行循环初始化
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight2[i], a=math.sqrt(5))

    def forward(self, x):
        # x shape: [Num_Experts, Capacity, In_Features]
        # Weight1: [Num_Experts, In_Features, Inter_Dim]
        
        # 1. 第一层并行计算 (Batch Matrix Multiply)
        # 结果: [Num_Experts, Capacity, Inter_Dim]
        h = torch.bmm(x, self.weight1)
        
        # 2. 激活函数
        if self.use_gate:
            gate, val = h.chunk(2, dim=-1)
            h = self.act(gate) * val
        else:
            h = self.act(h)
            
        h = self.dropout(h)
        
        # 3. 第二层并行计算
        # Weight2: [Num_Experts, Hidden_Features, Out_Features]
        out = torch.bmm(h, self.weight2)
        
        return out

class ParallelMoE(BasicModel):
    def __init__(
        self,
        model_dim: int,
        top_k: int,
        num_experts: int,
        num_shared_experts: int = 0,
        use_aux_loss: bool = False,
        use_gate: bool = False,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.use_aux_loss = use_aux_loss
        self.capacity_factor = capacity_factor
        self.use_gate = use_gate

        hidden_dim = model_dim * 4

        # 1. 门控网络
        self.gate = nn.Linear(model_dim, num_experts, bias=False)

        # 2. 并行专家 (替代原来的 ModuleList)
        self.parallel_experts = ParallelExpert(
            num_experts, model_dim, hidden_dim, model_dim, use_gate=use_gate
        )

        # 3. 共享专家 (稍微简单处理，也可以并行化，但通常共享专家数量少)
        self.shared_experts = None
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                self._create_mlp(model_dim, hidden_dim, use_gate) for _ in range(num_shared_experts)
            ])

    def _create_mlp(self, dim, hidden, use_gate):
        # 简单的 MLP 构建辅助函数
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                inter = hidden * 2 if use_gate else hidden
                self.fc1 = nn.Linear(dim, inter)
                self.fc2 = nn.Linear(hidden, dim)
                self.act = nn.SiLU() if use_gate else nn.GELU()
            def forward(self, x):
                h = self.fc1(x)
                if use_gate:
                    g, v = h.chunk(2, dim=-1)
                    h = self.act(g) * v
                else:
                    h = self.act(h)
                return self.fc2(h)
        return SimpleMLP()

    def count_params(self, trainable_only: bool = False, active_only: bool = False) -> int:
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
        total = self.count_params(active_only=False)
        active = self.count_params(active_only=True)
        return MoEInfo(total_count=total, active_count=active)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        """
        全并行的 Forward 流程
        """
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
