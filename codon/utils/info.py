import torch
import platform
import psutil
import socket
import json
import importlib.metadata
try:
    import numpy as np
except ImportError:
    np = None

from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class SystemEnvironment:
    '''
    Data class for storing a comprehensive snapshot of the experimental environment.

    Attributes:
        timestamp (str): The start time of the experiment.
        hostname (str): The hostname of the execution machine.
        os_info (Dict[str, str]): Detailed operating system version information.
        cpu_info (Dict[str, Any]): CPU model and number of cores.
        ram_total_gb (float): Total RAM size in GB.
        gpu_info (Dict[str, Any]): GPU model, memory, and CUDA/cuDNN versions.
        libraries (Dict[str, str]): Versions of core third-party libraries (NumPy, PyTorch, Triton, etc.).
        seeds (Dict[str, Any]): The runtime state of random seeds.
    '''
    timestamp: str
    hostname: str
    os_info: Dict[str, str]
    cpu_info: Dict[str, Any]
    ram_total_gb: float
    gpu_info: Dict[str, Any]
    libraries: Dict[str, str]
    seeds: Dict[str, Any]

    def to_json(self, indent: int = 4) -> str:
        '''
        Serializes the environment information into a JSON string.

        Args:
            indent (int): Number of spaces for indentation. Defaults to 4.

        Returns:
            str: Environment information in JSON format.
        '''
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def save(self, path: str):
        '''
        Saves the environment information to a file.

        Args:
            path (str): The path to save the file.
        '''
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

def get_system_info(manual_seed: Optional[int] = None) -> SystemEnvironment:
    '''
    Automatically collects the current system's hardware and software environment along with the random seed state.

    Args:
        manual_seed (Optional[int]): If you manually set a global seed at the beginning of the code,
            you can pass this parameter to ensure recording consistency.

    Returns:
        SystemEnvironment: The populated environment information object.
    '''
    import codon
    manual_seed = manual_seed if manual_seed is not None else codon.__seed__

    # Track core scientific computing and acceleration libraries
    target_libs = [
        'codon-model',
        'numpy', 'torch', 'triton', 'transformers', 'tokenizers',
        'accelerate', 'safetensors', 'scipy', 'pandas'
    ]
    lib_versions = {}
    for lib in target_libs:
        try:
            lib_versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            lib_versions[lib] = 'not_installed'

    # GPU and parallel computing environment detection
    gpu_devices = []
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_devices.append({
                'index': i,
                'name': props.name,
                'memory_mb': props.total_memory // (1024**2),
                'compute_capability': f'{props.major}.{props.minor}'
            })

    # Random seed state capture
    # Note: numpy can only capture the hash of the current state or None
    seeds_status = {
        'python_random': 'dynamic', 
        'torch_initial_seed': torch.initial_seed(),
        'torch_cuda_initial_seed': torch.cuda.initial_seed() if cuda_available else 'N/A',
        'manual_seed_provided': manual_seed
    }
    if np:
        # Record the hash of the numpy state to verify state consistency, not the raw seed integer
        seeds_status['numpy_state_hash'] = hash(str(np.random.get_state()))

    return SystemEnvironment(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        hostname=socket.gethostname(),
        os_info={
            'system': platform.system(),
            'version': platform.version(),
            'release': platform.release(),
            'architecture': platform.machine()
        },
        cpu_info={
            'model': platform.processor(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True)
        },
        ram_total_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        gpu_info={
            'count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda if cuda_available else 'N/A',
            'cudnn_version': str(torch.backends.cudnn.version()) if cuda_available else 'N/A',
            'devices': gpu_devices
        },
        libraries=lib_versions,
        seeds=seeds_status
    )
