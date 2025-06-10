"""
Test script to verify MPS backend support and functionality.
Run this to check if MPS is properly configured and working.
"""

import torch
import platform
import sys
from mps_utils import is_mps_available, configure_mps_environment, convert_dtype_for_mps


def test_mps_availability():
    """Test if MPS backend is available."""
    print("=" * 50)
    print("MPS Availability Test")
    print("=" * 50)
    
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Use our utility function
    print(f"MPS fully functional: {is_mps_available()}")
    print()


def test_mps_tensor_operations():
    """Test basic tensor operations on MPS."""
    if not is_mps_available():
        print("MPS not available, skipping tensor operation tests")
        return
    
    print("=" * 50)
    print("MPS Tensor Operations Test")
    print("=" * 50)
    
    try:
        # Test tensor creation
        print("Creating tensors on MPS...")
        x = torch.randn(3, 4, device="mps")
        y = torch.randn(4, 5, device="mps")
        print(f"✓ Tensor creation successful")
        
        # Test matrix multiplication
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful, result shape: {z.shape}")
        
        # Test dtype conversion
        x_float64 = torch.randn(3, 4, dtype=torch.float64)
        x_mps = convert_dtype_for_mps(x_float64).to("mps")
        print(f"✓ Dtype conversion successful: {x_float64.dtype} -> {x_mps.dtype}")
        
        # Test common operations
        operations = [
            ("Addition", lambda: x + x),
            ("Multiplication", lambda: x * 2),
            ("Softmax", lambda: torch.softmax(x, dim=-1)),
            ("Layer Norm", lambda: torch.nn.functional.layer_norm(x, x.shape[1:])),
            ("GELU activation", lambda: torch.nn.functional.gelu(x)),
        ]
        
        for op_name, op_func in operations:
            try:
                result = op_func()
                print(f"✓ {op_name} successful")
            except Exception as e:
                print(f"✗ {op_name} failed: {e}")
        
        # Test memory operations
        torch.mps.empty_cache()
        print(f"✓ MPS cache cleared successfully")
        
    except Exception as e:
        print(f"✗ MPS tensor operations failed: {e}")
    
    print()


def test_mps_model():
    """Test a simple model on MPS."""
    if not is_mps_available():
        print("MPS not available, skipping model tests")
        return
    
    print("=" * 50)
    print("MPS Model Test")
    print("=" * 50)
    
    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.Softmax(dim=-1)
        )
        
        # Move model to MPS
        model = model.to("mps")
        print("✓ Model moved to MPS successfully")
        
        # Test forward pass
        x = torch.randn(5, 10, device="mps")
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test different dtypes
        model_float32 = model.to(torch.float32)
        output_float32 = model_float32(x.to(torch.float32))
        print(f"✓ Float32 inference successful")
        
    except Exception as e:
        print(f"✗ MPS model test failed: {e}")
    
    print()


def test_mps_performance():
    """Simple performance comparison between CPU and MPS."""
    if not is_mps_available():
        print("MPS not available, skipping performance tests")
        return
    
    print("=" * 50)
    print("MPS Performance Test")
    print("=" * 50)
    
    import time
    
    # Matrix multiplication benchmark
    size = 1000
    iterations = 100
    
    # CPU benchmark
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start_time = time.time()
    for _ in range(iterations):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    
    # MPS benchmark
    x_mps = x_cpu.to("mps")
    y_mps = y_cpu.to("mps")
    torch.mps.synchronize()  # Ensure all operations are complete
    
    start_time = time.time()
    for _ in range(iterations):
        z_mps = torch.matmul(x_mps, y_mps)
    torch.mps.synchronize()  # Ensure all operations are complete
    mps_time = time.time() - start_time
    
    print(f"Matrix multiplication ({size}x{size}) - {iterations} iterations:")
    print(f"CPU time: {cpu_time:.3f}s")
    print(f"MPS time: {mps_time:.3f}s")
    print(f"Speedup: {cpu_time/mps_time:.2f}x")
    print()


def main():
    """Run all MPS tests."""
    print("\n🔍 Testing MPS Backend Support\n")
    
    # Configure MPS environment
    configure_mps_environment()
    
    # Run tests
    test_mps_availability()
    test_mps_tensor_operations()
    test_mps_model()
    test_mps_performance()
    
    print("✅ MPS testing complete!")


if __name__ == "__main__":
    main()