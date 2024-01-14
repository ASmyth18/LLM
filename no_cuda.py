import torch
import os

os.environ['CUDA_HOME'] = r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3'  # Update the path accordingly

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')


print(f'CUDA_HOME: {os.environ.get("CUDA_HOME")}')
print(f'PATH: {os.environ.get("PATH")}')
print(torch.__version__)