from codon import *


def _prepare_tensors(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    '''
    Helper function to flatten tensors and extract batch size for batch-wise computation.
    
    Args:
        input_tensor (torch.Tensor): Input data tensor.
        output_tensor (torch.Tensor): Output data tensor.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: Reshaped input (batch_size, in_features), 
                                                reshaped output (batch_size, out_features), 
                                                and batch_size.
    '''
    in_features = input_tensor.size(-1)
    out_features = output_tensor.size(-1)
    
    x = input_tensor.reshape(-1, in_features)
    y = output_tensor.reshape(-1, out_features)
    batch_size = x.size(0)
    
    return x, y, batch_size

def hebbian_update(weight: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using the basic Hebbian learning rule.
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    delta_w = learning_rate * torch.matmul(y.transpose(0, 1), x) / batch_size
    return delta_w

def oja_update(weight: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using Oja's rule, a normalized Hebbian rule.
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    hebbian_term = torch.matmul(y.transpose(0, 1), x) / batch_size
    
    y_sq = torch.mean(y ** 2, dim=0).view(y.size(-1), 1)
    forgetting_term = y_sq * weight
    
    delta_w = learning_rate * (hebbian_term - forgetting_term)
    return delta_w

def bcm_update(weight: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor, threshold: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using the Bienenstock-Cooper-Munro (BCM) rule.
    
    The BCM rule introduces a sliding threshold to determine whether a synapse should be 
    potentiated (LTP) or depressed (LTD).
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        threshold (torch.Tensor): Sliding threshold tensor of shape (out_features,).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    y_mod = y * (y - threshold.unsqueeze(0))
    
    delta_w = learning_rate * torch.matmul(y_mod.transpose(0, 1), x) / batch_size
    return delta_w

def covariance_update(weight: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using the Covariance learning rule (Sejnowski rule).
    
    This rule centers the input and output activations by subtracting their means, 
    allowing for both potentiation and depression.
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    delta_w = learning_rate * torch.matmul(y_centered.transpose(0, 1), x_centered) / batch_size
    return delta_w

def anti_hebbian_update(weight: torch.Tensor, output_tensor: torch.Tensor, target_rate: float = 0.0, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using the Anti-Hebbian learning rule (Foldiak's rule).
    
    Typically used for lateral inhibition. If two neurons fire together, their inhibitory 
    connection strength increases to enforce decorrelation/sparsity.
    
    Args:
        weight (torch.Tensor): Current lateral weight tensor of shape (out_features, out_features).
        output_tensor (torch.Tensor): Output/activation data tensor of shape (..., out_features).
        target_rate (float, optional): Desired average firing rate. Defaults to 0.0.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated lateral weight update.
    '''
    # To reuse _prepare_tensors which expects two arguments, we duplicate output_tensor
    _, y, batch_size = _prepare_tensors(output_tensor, output_tensor)
    
    y_centered = y - target_rate
    
    correlation = torch.matmul(y_centered.transpose(0, 1), y) / batch_size
    
    delta_w = learning_rate * (correlation - weight)
    delta_w.fill_diagonal_(0.0)
    
    return delta_w

def local_error_driven_update(error_tensor: torch.Tensor, state_tensor: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates weight updates driven by local prediction errors.
    
    This abstracts the weight update in Predictive Coding, where changes are proportional 
    to the product of presynaptic activity (state) and postsynaptic error.
    
    Args:
        error_tensor (torch.Tensor): Postsynaptic error tensor. Shape should align with the output dimension of weight.
        state_tensor (torch.Tensor): Presynaptic state tensor. Shape should align with the input dimension of weight.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    # Here state is input-like and err is output-like in terms of W matrix multiplication
    state, err, batch_size = _prepare_tensors(state_tensor, error_tensor)
    
    delta_w = learning_rate * torch.matmul(err.transpose(0, 1), state) / batch_size
    return delta_w

def synaptic_scaling_update(weight: torch.Tensor, output_tensor: torch.Tensor, target_rate: float = 0.1, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using Synaptic Scaling (Homeostatic Plasticity).
    
    Maintains the average firing rate of neurons within a target range by multiplicatively
    scaling the current weights.
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        output_tensor (torch.Tensor): Output/activation data tensor of shape (..., out_features).
        target_rate (float, optional): Desired average firing rate. Defaults to 0.1.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    _, y, _ = _prepare_tensors(output_tensor, output_tensor)
    
    y_mean = y.mean(dim=0).view(y.size(-1), 1)
    
    delta_w = learning_rate * weight * (target_rate - y_mean)
    return delta_w

def vogels_sprekeler_update(input_tensor: torch.Tensor, output_tensor: torch.Tensor, target_rate: float = 0.1, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using the Vogels-Sprekeler rule for Inhibitory-to-Excitatory balance.
    
    Adjusts inhibitory weights to track and cancel out excitatory inputs, maintaining
    the network at a critical balance state.
    
    Args:
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        target_rate (float, optional): Desired average firing rate. Defaults to 0.1.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update for inhibitory connections.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    y_mod = y - target_rate
    
    delta_w = learning_rate * torch.matmul(y_mod.transpose(0, 1), x) / batch_size
    return delta_w

def reward_modulated_hebbian_update(input_tensor: torch.Tensor, output_tensor: torch.Tensor, reward: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using Reward-Modulated Hebbian Learning (Three-Factor rule).
    
    Args:
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        reward (torch.Tensor): Global reward signal tensor of shape (..., 1) or scalar.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    if reward.dim() == 0 or reward.numel() == 1:
        r = reward.view(1, 1).expand(batch_size, 1)
    else:
        r = reward.reshape(-1, 1)
        
    y_r = y * r
    
    delta_w = learning_rate * torch.matmul(y_r.transpose(0, 1), x) / batch_size
    return delta_w

def eligibility_trace_update(trace: torch.Tensor, reward: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using an Eligibility Trace and a delayed reward.
    
    Args:
        trace (torch.Tensor): The accumulated eligibility trace tensor.
        reward (torch.Tensor): Global reward signal.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The actual weight update.
    '''
    if reward.dim() == 0 or reward.numel() == 1:
        r = reward.item()
    else:
        r = reward.mean().item()
        
    delta_w = learning_rate * r * trace
    return delta_w

def instar_update(weight: torch.Tensor, input_tensor: torch.Tensor, output_tensor: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using Grossberg's Instar rule.
    
    Weights move towards the input patterns that activate them, suitable for clustering
    and dictionary learning (often combined with Winner-Take-All).
    
    Args:
        weight (torch.Tensor): Current weight tensor of shape (out_features, in_features).
        input_tensor (torch.Tensor): Input data tensor of shape (..., in_features).
        output_tensor (torch.Tensor): Output data tensor of shape (..., out_features).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x, y, batch_size = _prepare_tensors(input_tensor, output_tensor)
    
    hebbian_term = torch.matmul(y.transpose(0, 1), x)
    forgetting_term = y.sum(dim=0).view(y.size(-1), 1) * weight
    
    delta_w = (learning_rate / batch_size) * (hebbian_term - forgetting_term)
    return delta_w

def rate_based_stdp_update(input_current: torch.Tensor, input_prev: torch.Tensor, output_current: torch.Tensor, output_prev: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    '''
    Calculates the weight update using a Rate-based STDP approximation.
    
    Captures temporal causality by using the discrete temporal derivatives of 
    inputs and outputs.
    
    Args:
        input_current (torch.Tensor): Current input data tensor of shape (..., in_features).
        input_prev (torch.Tensor): Previous input data tensor of shape (..., in_features).
        output_current (torch.Tensor): Current output data tensor of shape (..., out_features).
        output_prev (torch.Tensor): Previous output data tensor of shape (..., out_features).
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        
    Returns:
        torch.Tensor: The calculated weight update.
    '''
    x_t, y_t, batch_size = _prepare_tensors(input_current, output_current)
    x_tm1, y_tm1, _ = _prepare_tensors(input_prev, output_prev)
    
    dot_x = x_t - x_tm1
    dot_y = y_t - y_tm1
    
    term1 = torch.matmul(dot_y.transpose(0, 1), x_t)
    term2 = torch.matmul(y_t.transpose(0, 1), dot_x)
    
    delta_w = learning_rate * (term1 - term2) / batch_size
    return delta_w
