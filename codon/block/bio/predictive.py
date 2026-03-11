import torch.nn.functional as F

from codon.base    import *
from codon.ops.bio import (
    hebbian_update, 
    oja_update, 
    bcm_update, 
    covariance_update, 
    anti_hebbian_update, 
    local_error_driven_update,
    synaptic_scaling_update,
    vogels_sprekeler_update,
    reward_modulated_hebbian_update,
    eligibility_trace_update,
    instar_update,
    rate_based_stdp_update
)

from dataclasses import dataclass
from typing import Optional, Dict
import math


@dataclass
class PredictiveCodingOutput:
    '''
    Output of the PredictiveCoding layer with enhanced biological features.

    Attributes:
        output_tensor (torch.Tensor): The output/activation (state 'r') of the layer.
        predicted_input (torch.Tensor): The top-down prediction of the input.
        error_tensor (torch.Tensor): The precision-weighted prediction error.
        weight_updates (Dict[str, torch.Tensor]): A dictionary containing weight updates 
                                                  for various synapses (forward, feedback, lateral).
    '''
    output_tensor: torch.Tensor
    predicted_input: torch.Tensor
    error_tensor: torch.Tensor
    weight_updates: Dict[str, torch.Tensor]


class PredictiveCoding(BasicModel):
    '''
    An advanced Predictive Coding layer incorporating iterative inference, 
    separated feedback pathways, lateral inhibition, precision weighting, 
    and sparsity constraints.

    This module simulates a biological microcircuit that minimizes free energy 
    (prediction error) over time (inference steps) before updating its synaptic weights.

    Attributes:
        weight (nn.Parameter): Forward synaptic weights (driving connections).
        feedback_weight (nn.Parameter, optional): Feedback weights (predictive connections).
        lateral_weight (nn.Parameter, optional): Lateral inhibitory weights between neurons.
        precision (nn.Parameter, optional): Precision weighting for the error signal.
        bias (nn.Parameter, optional): Bias term for the forward activation.
    '''

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        learning_rate: float = 0.01, 
        rule: str = 'oja', 
        use_bias: bool = True,
        inference_steps: int = 20,
        inference_lr: float = 0.1,
        inference_decay: float = 1.0,
        separated_weight: bool = False,
        lateral_inhibition: bool = False,
        use_precision: bool = False,
        sparsity_weight: float = 0.0,
        auto_update: bool = False,
        bcm_momentum: float = 0.1,
        use_stp: bool = False,
        stp_u0: float = 0.2,
        stp_tau_u: float = 10.0,
        stp_tau_x: float = 10.0,
        target_rate: float = 0.1,
        trace_decay: float = 0.9
    ) -> None:
        '''
        Initializes the advanced PredictiveCoding layer.

        Args:
            in_features (int): Dimension of the input data.
            out_features (int): Dimension of the output representation.
            learning_rate (float, optional): Synaptic plasticity learning rate. Defaults to 0.01.
            rule (str, optional): Learning rule ('hebbian', 'oja', 'bcm', 'covariance'). Defaults to 'oja'.
            use_bias (bool, optional): Whether to use a bias term. Defaults to True.
            inference_steps (int, optional): Number of iterations for state inference. Defaults to 20.
            inference_lr (float, optional): Learning rate for state inference dynamics. Defaults to 0.1.
            inference_decay (float, optional): Decay factor applied to inference_lr per step. Defaults to 1.0.
            separated_weight (bool, optional): Use distinct weights for feedforward and feedback. Defaults to False.
            lateral_inhibition (bool, optional): Enable lateral competition between neurons. Defaults to False.
            use_precision (bool, optional): Learn precision (inverse variance) for errors. Defaults to False.
            sparsity_weight (bool, optional): L1 penalty coefficient for sparse coding. Defaults to 0.0.
            auto_update (bool, optional): Automatically apply calculated weight updates in forward. Defaults to False.
            bcm_momentum (float, optional): Momentum for BCM sliding threshold. Defaults to 0.1.
            use_stp (bool, optional): Enable Short-Term Plasticity (Tsodyks-Markram model). Defaults to False.
            stp_u0 (float, optional): Baseline release probability for STP. Defaults to 0.2.
            stp_tau_u (float, optional): Facilitation time constant. Defaults to 10.0.
            stp_tau_x (float, optional): Depression time constant. Defaults to 10.0.
            target_rate (float, optional): Desired average firing rate for homeostasis. Defaults to 0.1.
            trace_decay (float, optional): Decay factor for eligibility trace. Defaults to 0.9.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.rule = rule.lower()
        self.use_bias = use_bias
        
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.inference_decay = inference_decay
        self.separated_weight = separated_weight
        self.lateral_inhibition = lateral_inhibition
        self.use_precision = use_precision
        self.sparsity_weight = sparsity_weight
        self.auto_update = auto_update
        self.bcm_momentum = bcm_momentum
        
        self.use_stp = use_stp
        self.stp_u0 = stp_u0
        self.stp_tau_u = stp_tau_u
        self.stp_tau_x = stp_tau_x
        self.target_rate = target_rate
        self.trace_decay = trace_decay

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        
        if self.rule == 'bcm':
            self.register_buffer('bcm_threshold', torch.zeros(out_features))
        else:
            self.bcm_threshold = None
            
        if self.rule == 'stdp':
            self.register_buffer('prev_input', None)
            self.register_buffer('prev_state', None)
            
        if self.rule == 'eligibility':
            self.register_buffer('eligibility_trace', None)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        if self.separated_weight:
            self.feedback_weight = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        else:
            self.register_parameter('feedback_weight', None)
            
        if self.lateral_inhibition:
            self.lateral_weight = nn.Parameter(torch.Tensor(out_features, out_features), requires_grad=False)
        else:
            self.register_parameter('lateral_weight', None)
            
        if self.use_precision:
            self.precision = nn.Parameter(torch.Tensor(in_features), requires_grad=False)
        else:
            self.register_parameter('precision', None)
            
        if self.use_stp:
            self.register_buffer('stp_u_state', torch.tensor([stp_u0]))
            self.register_buffer('stp_x_state', torch.tensor([1.0]))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets all synaptic parameters using Kaiming/Uniform initializations.
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.precision is not None:
            nn.init.constant_(self.precision, 0.0)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        if self.feedback_weight is not None:
            nn.init.kaiming_uniform_(self.feedback_weight, a=math.sqrt(5))
            
        if self.lateral_weight is not None:
            nn.init.uniform_(self.lateral_weight, 0.0, 0.01)
            self.lateral_weight.data.fill_diagonal_(0.0)

    def _get_feedback_matrix(self) -> torch.Tensor:
        '''
        Helper to get the feedback matrix, handling the separated_weight logic.
        '''
        if self.separated_weight:
            return self.feedback_weight
        return self.weight.t()

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor, reward: Optional[torch.Tensor] = None) -> PredictiveCodingOutput:
        '''
        Performs iterative inference to find the optimal activation state,
        then calculates biologically plausible synaptic updates.

        Note:
            Decorated with @torch.no_grad() to block global backpropagation.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).
            reward (Optional[torch.Tensor], optional): Global reward signal for 'reward_hebb' rule. Defaults to None.

        Returns:
            PredictiveCodingOutput: Output containing the steady state, prediction, error, and updates.
        '''
        if self.use_stp:
            if self.stp_u_state.shape != input_tensor.shape:
                self.stp_u_state = torch.full_like(input_tensor, self.stp_u0)
                self.stp_x_state = torch.ones_like(input_tensor)
                
            u_new = self.stp_u_state + (self.stp_u0 - self.stp_u_state) / self.stp_tau_u + self.stp_u0 * (1.0 - self.stp_u_state) * input_tensor
            x_new = self.stp_x_state + (1.0 - self.stp_x_state) / self.stp_tau_x - self.stp_u_state * self.stp_x_state * input_tensor
            
            self.stp_u_state = u_new.clamp(0.0, 1.0).detach()
            self.stp_x_state = x_new.clamp(0.0, 1.0).detach()
            
            effective_input = input_tensor * self.stp_u_state * self.stp_x_state
        else:
            effective_input = input_tensor
        
        r_state = F.linear(effective_input, self.weight, self.bias)
        current_inf_lr = self.inference_lr
        
        W_fb = self._get_feedback_matrix()
        
        for _ in range(self.inference_steps):
            x_hat = F.linear(r_state, W_fb.t())
            raw_error = input_tensor - x_hat
            
            if self.use_precision:
                pi = torch.exp(self.precision)
                error_signal = pi * raw_error
            else:
                error_signal = raw_error
                
            state_grad = F.linear(error_signal, self.weight)
                
            if self.lateral_inhibition:
                lat_w = self.lateral_weight.clone()
                lat_w.fill_diagonal_(0.0)
                inhibition = F.linear(r_state, lat_w)
                state_grad = state_grad - inhibition
                
            if self.sparsity_weight > 0.0:
                state_grad = state_grad - self.sparsity_weight * torch.sign(r_state)
                
            r_state = r_state + current_inf_lr * state_grad
            r_state = F.relu(r_state)
            current_inf_lr *= self.inference_decay

        updates: Dict[str, torch.Tensor] = {}
        
        final_x_hat = F.linear(r_state, W_fb.t())
        final_error = input_tensor - final_x_hat
        
        if self.use_precision:
            pi = torch.exp(self.precision)
            final_error = pi * final_error
            
        if self.rule == 'hebbian':
            updates['weight'] = hebbian_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'oja':
            updates['weight'] = oja_update(self.weight, input_tensor, r_state, self.learning_rate)
        elif self.rule == 'bcm':
            updates['weight'] = bcm_update(self.weight, input_tensor, r_state, self.bcm_threshold, self.learning_rate)
            # Update sliding threshold: E[y^2]
            with torch.no_grad():
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
            raise ValueError(f"Unsupported learning rule: {self.rule}")
            
        if self.separated_weight:
            updates['feedback_weight'] = local_error_driven_update(final_error, r_state, self.learning_rate)
            
        if self.lateral_inhibition:
            updates['lateral_weight'] = anti_hebbian_update(self.lateral_weight, r_state, target_rate=self.target_rate, learning_rate=self.learning_rate)
            
        if self.use_precision:
            sq_error_mean = torch.mean(raw_error ** 2, dim=0)
            target_log_precision = -torch.log(sq_error_mean + 1e-8)
            updates['precision'] = self.learning_rate * (target_log_precision - self.precision)

        if self.auto_update:
            self.apply_updates(updates)

        return PredictiveCodingOutput(
            output_tensor=r_state,
            predicted_input=final_x_hat,
            error_tensor=final_error,
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
            
        if 'feedback_weight' in updates and getattr(self, 'feedback_weight', None) is not None:
            self.feedback_weight.add_(updates['feedback_weight'])
            
        if 'lateral_weight' in updates and getattr(self, 'lateral_weight', None) is not None:
            self.lateral_weight.add_(updates['lateral_weight'])
            self.lateral_weight.clamp_(min=0.0)
            self.lateral_weight.fill_diagonal_(0.0)
            
        if 'precision' in updates and getattr(self, 'precision', None) is not None:
            self.precision.add_(updates['precision'])

    def reconstruct(self, state_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Reconstructs the expected top-down input from a given activation state.

        This method acts as the generative pathway of the predictive coding circuit.
        It utilizes the feedback connections (either tied to feedforward weights 
        or separated) to generate the prediction of the original input.

        Args:
            state_tensor (torch.Tensor): The latent activation state, typically 
                                         with shape (batch_size, out_features).

        Returns:
            torch.Tensor: The reconstructed/predicted input with shape 
                          (batch_size, in_features).
        '''
        feedback_matrix = self._get_feedback_matrix()
        reconstructed_input = F.linear(state_tensor, feedback_matrix.t())
        return reconstructed_input
