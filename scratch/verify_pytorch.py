"""

"""
import sys
import torch

def verify_pytorch():
    print("=" * 60)
    print("PYTORCH ENVIRONMENT & TENSOR VERIFICATION TEST")
    print("=" * 60)

    # System & Environment Specs
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_avail else 0
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Default)"

    print(f"Python Version:   {python_ver}")
    print(f"PyTorch Version:  {torch_ver}")
    print(f"CPU Available:    True")
    print(f"CUDA Available:   {cuda_avail}")
    print(f"GPU Device Count: {device_count}")
    print(f"Active Device:    {device_name}")

    # Tensor Operation Test
    print("\nRunning PyTorch Tensor Operations Test...")
    x = torch.rand(5, 3)
    y = torch.ones(5, 3)
    z = x + y

    print(f"  - Random Tensor x shape: {x.shape}")
    print(f"  - Ones Tensor y shape:   {y.shape}")
    print(f"  - Sum Tensor z (x + y):\n{z}")
    
    # Check autograd computation graph
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = a ** 2 + 3 * a
    out = b.sum()
    out.backward()
    print(f"  - Autograd test passed (gradients: {a.grad.tolist()})")

    print("=" * 60)
    print("PYTORCH TENSOR TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    verify_pytorch()
