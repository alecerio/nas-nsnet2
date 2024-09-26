from nsnet2.pytorch.nsnet2.nsnet2 import NsNet2_npy
from nsnet2.pytorch.nsnet2_q.nsnet2_quantized import Q_NsNet2_npy
import numpy as np
import torch

torch.manual_seed(42)
x = torch.randn(1, 1, 257).numpy()*0.001
h1 = torch.randn(1, 1, 400).numpy()*0.001
h2 = torch.randn(1, 1, 400).numpy()*0.001

print(min(x.squeeze()))
print(max(x.squeeze()))

numpy_weights_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
baseline = NsNet2_npy(numpy_weights_path)
y = baseline(x, h1, h2)

quantized = Q_NsNet2_npy(numpy_weights_path)
yq = quantized(x, h1, h2)
