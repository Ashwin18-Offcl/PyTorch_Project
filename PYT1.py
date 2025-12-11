import torch

print(torch.__version__)
print(torch.rand(2, 3))
print("CUDA available:", torch.cuda.is_available())

