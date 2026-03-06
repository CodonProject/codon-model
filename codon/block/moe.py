import torch.nn.functional as F

from codon.base  import *
from dataclasses import dataclass
from typing      import Union

from codon.block.mlp import MLP


@dataclass
class MoEOutput:
    '''
    Output of the Mixture-of-Experts (MoE) model.

    Args:
        output (torch.Tensor): The final output tensor.
        aux_loss (Union[torch.Tensor, None]): The auxiliary loss for load balancing.
    '''
    output: torch.Tensor
    aux_loss: Union[torch.Tensor, None]

@dataclass
class MoEInfo:
    '''
    Parameter counts for the MoE model.

    Args:
        total_count (int): The total number of parameters.
        active_count (int): The number of active parameters.
    '''
    total_count: int
    active_count: int


class Expert(MLP):
    '''
    A single expert module in the Mixture-of-Experts architecture.
    Wraps the MLP module to provide consistent interface and default behavior.

    Attributes:
        See codon.block.mlp.MLP for attributes.
    '''
    def __init__(self, in_features: int, hidden_features: int, out_features: int, use_gate: bool = False, dropout: float = 0.1):
        '''
        Initialize the Expert module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            use_gate (bool, optional): Whether to use Gated MLP (e.g. SwiGLU). Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        '''
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            use_gate=use_gate,
            dropout=dropout
        )


class MoE(BasicModel):
    '''
    Mixture-of-Experts (MoE) module.
    Includes support for shared experts and auxiliary loss for load balancing.

    Attributes:
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.ModuleList, optional): List of shared expert modules.
        gate (nn.Linear): Gating network to route tokens to experts.
        use_gate (bool): Whether experts use gating mechanism.
    '''
    def __init__(
        self,
        model_dim: int,
        top_k: int,
        num_experts: int,
        num_shared_experts: int = 0,
        use_aux_loss: bool = False,
        use_gate: bool = False
    ):
        '''
        Initialize the MoE module.

        Args:
            model_dim (int): The dimension of the model.
            top_k (int): Number of experts to route to for each token.
            num_experts (int): Total number of experts.
            num_shared_experts (int, optional): Number of shared experts. Defaults to 0.
            use_aux_loss (bool, optional): Whether to use auxiliary loss for load balancing. Defaults to False.
            use_gate (bool, optional): Whether to use Gated MLP for experts. Defaults to False.
        '''
        super().__init__()
        self.model_dim = model_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.use_aux_loss = use_aux_loss
        self.use_gate = use_gate

        hidden_dim = model_dim * 4

        self.experts = nn.ModuleList([
            Expert(model_dim, hidden_dim, model_dim, use_gate=use_gate) for _ in range(num_experts)
        ])

        self.shared_experts = None
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(model_dim, hidden_dim, model_dim, use_gate=use_gate) for _ in range(num_shared_experts)
            ])
        
        self.gate = nn.Linear(model_dim, num_experts, bias=False)
    
    def count_params(self, trainable_only:bool=False, active_only:bool=False) -> int:
        '''
        Count the number of parameters in the model.
        
        Args:
            trainable_only (bool, optional): If True, count only trainable parameters.
            active_only (bool, optional): If True, count only active parameters (e.g. for MoE).
            
        Returns:
            int: The total number of parameters.
        '''
        if not active_only:
            return super().count_params(trainable_only, active_only)
        
        total = BasicModel._count_params_recursive(self.gate, trainable_only, active_only)
        
        if self.shared_experts is not None:
            total += BasicModel._count_params_recursive(self.shared_experts, trainable_only, active_only)
            
        if len(self.experts) > 0:
            single_expert_params = self.experts[0].count_params(trainable_only, active_only)
            total += single_expert_params * self.top_k
            
        return total

    @property
    def info(self) -> MoEInfo:
        '''
        Get parameter count information for the MoE module.

        Returns:
            MoEInfo: An object containing total and active parameter counts.
        '''
        def get_params_count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        gate_params = get_params_count(self.gate)

        shared_params = 0
        if self.shared_experts is not None:
            shared_params = get_params_count(self.shared_experts)

        expert_params = 0
        if len(self.experts) > 0:
            expert_params = get_params_count(self.experts[0])

        total_routed_params = get_params_count(self.experts)
        
        active_routed_params = expert_params * self.top_k

        total_count = gate_params + shared_params + total_routed_params
        active_count = gate_params + shared_params + active_routed_params

        return MoEInfo(total_count=total_count, active_count=active_count)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        '''
        Forward pass of the MoE module.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Seq, Dim].

        Returns:
            MoEOutput: Object containing the output tensor and optional auxiliary loss.
        '''
        original_shape = x.shape
        # [Batch, Seq, Dim] -> [Total_Tokens, Dim]
        x_flat = x.reshape(-1, self.model_dim)

        shared_output = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for expert in self.shared_experts: shared_output = shared_output + expert(x_flat)
        
        # logits: [Total_Tokens, Num_Experts]
        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)

        # topk_weights: [Total_Tokens, Top_K]
        # topk_indices: [Total_Tokens, Top_K]
        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Auxiliary Loss / Load Balancing Loss
        aux_loss = None
        if self.use_aux_loss and self.training:
            density_1 = routing_probs.mean(dim=0)
            
            mask = torch.zeros_like(routing_probs)
            mask.scatter_(1, topk_indices, 1.0)
            density_1_proxy = mask.mean(dim=0)
            
            aux_loss = (self.num_experts * (density_1 * density_1_proxy).sum())

        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_choice = torch.where(topk_indices == i)
            
            if batch_idx.numel() == 0: continue
            # [Num_Selected, Model_Dim]
            current_inputs = x_flat[batch_idx]
            current_expert_output = expert(current_inputs)
            # [Num_Selected, 1]
            current_weights = topk_weights[batch_idx, nth_choice].unsqueeze(-1)
            routed_output.index_add_(0, batch_idx, current_expert_output * current_weights)
            
        final_output = routed_output + shared_output
        # [Batch, Seq, Dim]
        final_output = final_output.reshape(original_shape)
        return MoEOutput(output=final_output, aux_loss=aux_loss)
