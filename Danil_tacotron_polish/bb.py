import torch

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if PyTorch was compiled with GPU support
print("PyTorch with GPU support:", torch.version.cuda is not None)
