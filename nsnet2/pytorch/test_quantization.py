from nsnet2 import NsNet2_npy
from nsnet2_quantized import Q_NsNet2_npy
import numpy as np
import torch

torch.manual_seed(42)
x = torch.randn(1, 1, 257).numpy()*0.001
h1 = torch.randn(1, 1, 400).numpy()
h2 = torch.randn(1, 1, 400).numpy()

print(min(x.squeeze()))
print(max(x.squeeze()))

baseline = NsNet2_npy()
y = baseline(x, h1, h2)

quantized = Q_NsNet2_npy()
yq = quantized(x, h1, h2)
