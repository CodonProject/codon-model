from codon import *


@torch.compile(fullgraph=True)
def muon_kernel_step(
    p_data, grad, buf, lr, momentum, nesterov, ns_steps, scale
):
    '''
    Core kernel step for Muon optimizer.

    Performs the Newton-Schulz iteration for orthogonalization and updates the parameter tensor.

    Args:
        p_data: Parameter tensor to be updated in-place.
        grad: Gradient tensor for the parameter.
        buf: Momentum buffer tensor.
        lr: Learning rate.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        scale: Scaling factor based on matrix dimensions.
    '''
    original_shape = grad.shape
    
    if len(original_shape) > 2:
        grad = grad.view(original_shape[0], -1)
    
    buf.mul_(momentum).add_(grad)
    g = grad.add(buf, alpha=momentum) if nesterov else buf

    M, N = g.shape
    transpose = M > N
    if transpose:
        g = g.T
        
    X = g.to(torch.float32)
    X = X / (X.norm() + 1e-7)
    
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for _ in range(ns_steps):
        XXT = X @ X.T
        XXT2 = XXT @ XXT
        X = torch.addmm(X, b * XXT + c * XXT2, X, beta=a, alpha=1.0)
        
    if transpose:
        X = X.T
        
    g_orth = X.to(grad.dtype)
    
    if len(original_shape) > 2:
        g_orth = g_orth.view(original_shape)
        
    # Apply the update
    p_data.add_(g_orth, alpha=-lr * scale)


class Muon(BasicOptimizer):
    '''
    Muon Optimizer.

    A momentum-based optimizer that uses Newton-Schulz orthogonalization to improve
    convergence. Designed for large-scale training with 2D (linear) and 4D (convolutional)
    weight matrices.

    The optimizer maintains a momentum buffer and applies orthogonalization to the
    gradient before updating parameters, which helps maintain stable updates and
    improves conditioning.

    Args:
        params: Iterable of parameters to optimize.
        lr (float): Learning rate. Default: 1e-3.
        momentum (float): Momentum factor. Must be in [0, 1). Default: 0.95.
        nesterov (bool): Enables Nesterov momentum. Default: True.
        ns_steps (int): Number of Newton-Schulz iterations for orthogonalization.
                       Higher values give better approximation but cost more. Default: 5.

    Raises:
        ValueError: If learning rate is negative or momentum is outside [0, 1).
        ValueError: If gradient tensor has fewer than 2 dimensions (Muon only supports 2D/4D weights).
    '''

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, *args, **kwargs):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
            
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        '''
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and
                                          returns the loss. Used for LBFGS-style
                                          optimizers but included for compatibility.

        Returns:
            torch.Tensor or None: The loss value if closure is provided, otherwise None.
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if len(grad.shape) < 2:
                    raise ValueError('Muon only supports 2D/4D weights.')
                
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    flat_shape = (grad.shape[0], -1) if len(grad.shape) > 2 else grad.shape
                    state['momentum_buffer'] = torch.zeros(flat_shape, dtype=grad.dtype, device=grad.device)
                
                buf = state['momentum_buffer']
                
                scale = max(grad.shape[0], grad.shape[1] if len(grad.shape) > 1 else 1) ** 0.5
                
                muon_kernel_step(
                    p_data=p.data,
                    grad=grad,
                    buf=buf,
                    lr=lr,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                    scale=scale
                )
                
        return loss