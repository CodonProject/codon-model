from codon import *


def complex_relu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if x.is_complex():
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    return F.relu(x, inplace=inplace)

def complex_silu(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return torch.complex(F.silu(x.real), F.silu(x.imag))
    return F.silu(x)

def complex_sigmoid(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return torch.complex(torch.sigmoid(x.real), torch.sigmoid(x.imag))
    return torch.sigmoid(x)

def mod_relu(x: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_complex(): return F.relu(x + bias)
    
    norm  = torch.abs(x)
    scale = F.relu(norm + bias) / (norm + eps)
    
    return x * scale

def mod_silu(x: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_complex(): 
        return F.silu(x + bias)
    
    norm  = torch.abs(x)
    scale = F.silu(norm + bias) / (norm + eps)

    return x * scale

def mod_sigmoid(x: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_complex():
        return torch.sigmoid(x + bias)
    
    norm = torch.abs(x)
    scale = torch.sigmoid(norm + bias) / (norm + eps)

    return x * scale