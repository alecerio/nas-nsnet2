from nsnet2 import NsNet2_npy
from nsnet2_quantized import Q_NsNet2_npy
import torch

torch.manual_seed(42)
x = torch.randn(1, 1, 257)
h1 = torch.randn(1, 1, 400)
h2 = torch.randn(1, 1, 400)

print(min(x.squeeze()))
print(max(x.squeeze()))

baseline = NsNet2_npy()
y = baseline(x, h1, h2)

quantized = Q_NsNet2_npy()
yq = quantized(x, h1, h2)

#print(y)
