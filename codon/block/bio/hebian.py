import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Dict
from dataclasses import dataclass

from codon.base import BasicModel
from codon.ops.bio import (
    hebbian_update, 
    oja_update, 
    bcm_update, 
    covariance_update, 
    instar_update,
    synaptic_scaling_update,
    vogels_sprekeler_update,
    reward_modulated_hebbian_update,
    rate_based_stdp_update,
    eligibility_trace_update
)

@dataclass
class HebianOutput:
    '''
    Output of the Hebian layer.

    Attributes:
        output_tensor (torch.Tensor): The output/activation of the layer.
        weight_updates (Dict[str, torch.Tensor]): A dictionary containing weight updates 
                                                  for synapses.
    '''
    output_tensor: torch.Tensor
    weight_updates: Dict[str, torch.Tensor]


class Hebian(BasicModel):
    '''
    A layer implementing various biologically plausible Hebbian learning rules.

    Attributes:
        weight (nn.Parameter): Forward synaptic weights.
        bias (nn.Parameter, optional): Bias term for the forward activation.
    '''

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        learning_rate: float = 0.01, 
        rule: str = 'oja', 
        use_bias: bool = True,
        auto_update: bool = False,
        bcm_momentum: float = 0.1,
        target_rate: float = 0.1,
        trace_decay: float = 0.9,
        activation: str = 'linear'
    ) -> None:
        '''
        Initializes the Hebian layer.

        Args:
            in_features (int): Dimension of the input data.
            out_features (int): Dimension of the output representation.
            learning_rate (float, optional): Synaptic plasticity learning rate. Defaults to 0.01.
            rule (str, optional): Learning rule ('hebbian', 'oja', 'bcm', 'covariance', 'instar', 
                                  'scaling', 'vogels', 'reward_hebb', 'stdp', 'eligibility'). Defaults to 'oja'.
            use_bias (bool, optional): Whether to use a bias term. Defaults to True.
            auto_update (bool, optional): Automatically apply calculated weight updates in forward. Defaults to False.
            bcm_momentum (float, optional): Momentum for BCM sliding threshold. Defaults to 0.1.
            target_rate (float, optional): Desired average firing rate for homeostasis. Defaults to 0.1.
            trace_decay (float, optional): Decay factor for eligibility trace. Defaults to 0.9.
            activation (str, optional): Activation function ('linear', 'relu', 'sigmoid', 'tanh'). Defaults to 'linear'.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.rule = rule.lower()
        self.use_bias = use_bias
        self.auto_update = auto_update
        self.bcm_momentum = bcm_momentum
        self.target_rate = target_rate
        self.trace_decay = trace_decay
        self.activation = activation.lower()

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        if self.rule == 'bcm':
            self.register_buffer('bcm_threshold', torch.zeros(out_features))
        else:
            self.bcm_threshold = None
            
        if self.rule == 'stdp':
            self.register_buffer('prev_input', None)
            self.register_buffer('prev_state', None)
            
        if self.rule == 'eligibility':
            self.register_buffer('eligibility_trace', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets all synaptic parameters using Kaiming/Uniform initializations.
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _apply_activation(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Applies the selected activation function.
        
        Args:
            input_tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        '''
        if self.activation == 'relu':
            return F.relu(input_tensor)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(input_tensor)
        elif self.activation == 'tanh':
            return torch.tanh(input_tensor)
        return input_tensor

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, reward: Optional[torch.Tensor] = None) -> HebianOutput:
        '''
        Calculates the forward pass and biological synaptic updates.

        Note:
            Decorated with @torch.no_grad() to block global backpropagation, 
            as learning is done via local plasticity rules.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).
            reward (Optional[torch.Tensor], optional): Global reward signal for 'reward_hebb' or 'eligibility' rules. Defaults to None.

        Returns:
            HebianOutput: Output containing the activation state and weight updates.
        '''
        r_state = F.linear(input_tensor, self.weight, self.bias)
        r_state = self._apply_activation(r_state)
        
        updates: Dict[str, torch.Tensor] = {}
        
        if self.rule == 'hebbian':
            updates['weight'] = hebbian_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'oja':
            updates['weight'] = oja_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'bcm':
            updates['weight'] = bcm_update(self.weight, input_tensor, r_state, self.bcm_threshold, self.learning_rate)
            # Update sliding threshold: E[y^2]
            current_y2 = torch.mean(r_state ** 2, dim=0)
            self.bcm_threshold.mul_(1 - self.bcm_momentum).add_(current_y2, alpha=self.bcm_momentum)
        elif self.rule == 'covariance':
            updates['weight'] = covariance_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'instar':
            updates['weight'] = instar_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'scaling':
            updates['weight'] = synaptic_scaling_update(self.weight, r_state, target_rate=self.target_rate, learning_rate=self.learning_rate)
        elif self.rule == 'vogels':
            updates['weight'] = vogels_sprekeler_update(input_tensor, r_state, target_rate=self.target_rate, learning_rate=self.learning_rate)
        elif self.rule == 'reward_hebb':
            if reward is None:
                raise ValueError("The 'reward_hebb' rule requires a reward signal to be passed to forward().")
            updates['weight'] = reward_modulated_hebbian_update(input_tensor, r_state, reward, self.learning_rate)
        elif self.rule == 'stdp':
            if getattr(self, 'prev_input', None) is None or self.prev_input.shape != input_tensor.shape:
                self.prev_input = input_tensor.clone().detach()
                self.prev_state = r_state.clone().detach()
                updates['weight'] = torch.zeros_like(self.weight)
            else:
                updates['weight'] = rate_based_stdp_update(input_tensor, self.prev_input, r_state, self.prev_state, self.learning_rate)
                self.prev_input = input_tensor.clone().detach()
                self.prev_state = r_state.clone().detach()
        elif self.rule == 'eligibility':
            current_hebbian = hebbian_update(self.weight, input_tensor, r_state, learning_rate=1.0)
            
            if getattr(self, 'eligibility_trace', None) is None or self.eligibility_trace.shape != self.weight.shape:
                self.eligibility_trace = torch.zeros_like(self.weight)
                
            self.eligibility_trace = self.eligibility_trace * self.trace_decay + current_hebbian
            
            if reward is not None:
                updates['weight'] = eligibility_trace_update(self.eligibility_trace, reward, self.learning_rate)
            else:
                updates['weight'] = torch.zeros_like(self.weight)
        else:
            raise ValueError(f'Unsupported learning rule: {self.rule}')

        if self.auto_update:
            self.apply_updates(updates)

        return HebianOutput(
            output_tensor=r_state,
            weight_updates=updates
        )

    @torch.no_grad()
    def apply_updates(self, updates: Dict[str, torch.Tensor]) -> None:
        '''
        Applies the calculated weight updates to the layer's parameters in-place.

        Args:
            updates (Dict[str, torch.Tensor]): A dictionary of parameter names and their updates.
        '''
        if 'weight' in updates and self.weight is not None:
            self.weight.add_(updates['weight'])
        
        if 'bias' in updates and getattr(self, 'bias', None) is not None:
            self.bias.add_(updates['bias'])
