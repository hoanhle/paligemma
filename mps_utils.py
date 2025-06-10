"""
MPS (Metal Performance Shaders) backend utilities for PyTorch.
Provides optimizations and compatibility functions for Apple Silicon GPUs.
"""

import torch
import os
import warnings
from typing import Optional, Union, Tuple


def is_mps_available() -> bool:
    """
    Check if MPS backend is available and properly configured.
    
    Returns:
        bool: True if MPS is available and usable, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False
    
    if not torch.backends.mps.is_built():
        warnings.warn("MPS backend is available but not built. Please reinstall PyTorch with MPS support.")
        return False
    
    # Try to create a tensor on MPS to verify it's working
    try:
        test_tensor = torch.zeros(1, device="mps")
        del test_tensor
        return True
    except Exception as e:
        warnings.warn(f"MPS backend test failed: {e}")
        return False


def configure_mps_environment():
    """
    Configure environment variables for optimal MPS performance.
    """
    # Set memory management settings
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    
    # Enable fallback for unsupported operations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Disable MPS profiling for better performance in production
    os.environ["PYTORCH_MPS_PROFILING"] = "0"


def optimize_mps_memory():
    """
    Optimize MPS memory usage by clearing caches and synchronizing.
    """
    if torch.backends.mps.is_available():
        # Empty MPS cache
        torch.mps.empty_cache()
        
        # Synchronize MPS operations
        torch.mps.synchronize()


def convert_dtype_for_mps(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor dtype to MPS-compatible format.
    MPS doesn't support float64, so convert to float32.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor with MPS-compatible dtype
    """
    if tensor.dtype == torch.float64:
        return tensor.to(torch.float32)
    elif tensor.dtype == torch.int64 and tensor.device.type == "mps":
        # MPS has better performance with int32 for some operations
        return tensor.to(torch.int32)
    return tensor


def safe_to_mps(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Safely move tensor to MPS device with proper dtype conversion.
    
    Args:
        tensor: Input tensor
        dtype: Target dtype (optional)
        
    Returns:
        Tensor on MPS device with appropriate dtype
    """
    if dtype is None:
        dtype = tensor.dtype
    
    # Convert dtype if necessary
    if dtype == torch.float64:
        dtype = torch.float32
    
    try:
        return tensor.to(device="mps", dtype=dtype)
    except Exception as e:
        warnings.warn(f"Failed to move tensor to MPS: {e}. Falling back to CPU.")
        return tensor.to(dtype=dtype)


def get_mps_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a generator for MPS device with optional seed.
    
    Args:
        seed: Random seed (optional)
        
    Returns:
        torch.Generator for MPS device
    """
    generator = torch.Generator(device="mps")
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def check_mps_operation_support(operation_name: str) -> bool:
    """
    Check if a specific operation is supported on MPS backend.
    
    Args:
        operation_name: Name of the PyTorch operation
        
    Returns:
        bool: True if operation is likely supported, False otherwise
    """
    # List of operations that may not be fully supported on MPS
    unsupported_ops = [
        "torch.complex64",
        "torch.complex128",
        "torch.bfloat16",
        # Add more as discovered
    ]
    
    return operation_name not in unsupported_ops


def wrap_mps_autocast(func):
    """
    Decorator to handle autocast for MPS operations.
    MPS doesn't support autocast like CUDA, so this provides a workaround.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with MPS-safe autocast handling
    """
    def wrapper(*args, **kwargs):
        # Get the current device from args if possible
        device = None
        for arg in args:
            if hasattr(arg, 'device'):
                device = arg.device
                break
        
        if device and device.type == "mps":
            # Run without autocast for MPS
            return func(*args, **kwargs)
        else:
            # Use autocast for CUDA/CPU
            device_type = "cuda" if device and device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                return func(*args, **kwargs)
    
    return wrapper


class MPSConfig:
    """Configuration class for MPS-specific settings."""
    
    def __init__(
        self,
        memory_fraction: float = 0.7,
        enable_fallback: bool = True,
        enable_profiling: bool = False,
        preferred_dtype: torch.dtype = torch.float32,
    ):
        self.memory_fraction = memory_fraction
        self.enable_fallback = enable_fallback  
        self.enable_profiling = enable_profiling
        self.preferred_dtype = preferred_dtype
    
    def apply(self):
        """Apply the MPS configuration."""
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(self.memory_fraction)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if self.enable_fallback else "0"
        os.environ["PYTORCH_MPS_PROFILING"] = "1" if self.enable_profiling else "0"


# Example usage function
def setup_mps_for_model(model: torch.nn.Module, config: Optional[MPSConfig] = None) -> torch.nn.Module:
    """
    Setup a model for optimal MPS performance.
    
    Args:
        model: PyTorch model
        config: MPS configuration (optional)
        
    Returns:
        Model configured for MPS
    """
    if config is None:
        config = MPSConfig()
    
    config.apply()
    configure_mps_environment()
    
    # Move model to MPS and convert dtypes
    model = model.to("mps")
    
    # Convert model parameters to preferred dtype
    if config.preferred_dtype != torch.float64:
        model = model.to(config.preferred_dtype)
    
    return model