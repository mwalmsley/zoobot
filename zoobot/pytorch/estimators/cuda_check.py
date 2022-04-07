import torch

print('Pytorch', torch.__version__)
print('CUDA', torch.cuda.is_available())
print('CUDA version', torch.version.cuda)
