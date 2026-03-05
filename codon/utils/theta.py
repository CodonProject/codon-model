import math
from dataclasses import dataclass

@dataclass
class ValidateRoPEConfig:
    '''
    Configuration result for RoPE/Sinusoidal validation.

    Attributes:
        is_passed (bool): Whether the configuration is valid.
        info (str): Assessment status.
        suggested_base (float): The recommended base value.
    '''
    is_passed: bool
    info: str
    suggested_base: float

def validate_rope_config(max_len: int, base: float) -> ValidateRoPEConfig:
    '''
    Validates if the RoPE/Sinusoidal base is sufficient for the given maximum length.

    Args:
        max_len (int): The maximum sequence length.
        base (float): The base value for RoPE.

    Returns:
        ValidateRoPEConfig: The validation result containing status and recommendation.
    '''
    max_period = 2 * math.pi * base
    
    if max_len <= 8192:
        recommended = 10000.0
    else:
        # base = 10000 * (scaling_factor ^ 1.1)
        scaling_factor = max_len / 4096
        recommended = 10000.0 * (scaling_factor ** 1.1)
        
    if recommended > 1000000:
        recommended = round(recommended / 1000000) * 1000000.0
    elif recommended > 100000:
        recommended = round(recommended / 100000) * 100000.0
    else:
        recommended = round(recommended / 10000) * 10000.0

    if max_period < max_len:
        return ValidateRoPEConfig(is_passed=False, info='critical_low', suggested_base=recommended)
    
    elif max_period < max_len * 2:
        return ValidateRoPEConfig(is_passed=False, info='low', suggested_base=recommended)
    
    elif max_period > max_len * 100:
        return ValidateRoPEConfig(is_passed=True, info='too_high', suggested_base=recommended)
    
    else:
        return ValidateRoPEConfig(is_passed=True, info='optimal', suggested_base=recommended)
