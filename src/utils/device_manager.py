import torch
from typing import Union


class DeviceManager:
    """Manages device allocation for training and inference."""
    
    def __init__(self, device: str = 'cuda', num_gpus: int = 1):
        """
        Initialize device manager.
        
        Args:
            device: 'cuda', 'cpu', or 'mps' (for Apple Silicon)
            num_gpus: Number of GPUs to use (for multi-GPU training)
        """
        self.device_type = device
        self.num_gpus = num_gpus
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get appropriate device."""
        if self.device_type == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
        elif self.device_type == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("MPS requested but not available, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]):
        """Move tensor or model to device."""
        return tensor_or_model.to(self.device)
    
    def get_device_info(self) -> dict:
        """Get information about current device."""
        info = {
            'device_type': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_allocated': torch.cuda.memory_allocated(0),
                'cuda_memory_cached': torch.cuda.memory_reserved(0),
            })
        
        return info
