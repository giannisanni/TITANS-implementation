import torch
import requests

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Requests version: {requests.__version__}")
