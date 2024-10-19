from nsnet2.pytorch.nsnet2.nsnet2 import NsNet2_npy
from nsnet2.pytorch.nsnet2_q.nsnet2_quantized import Q_NsNet2_npy
from objective_function.inference_time.compute_inference_time import compute_inference_time
from objective_function.memory_footprint.compute_memory_footprint import compute_memory_footprint
from objective_function.pesq.compute_pesq import compute_pesq
from objective_function.objf.compute_objective_function import compute_objective_function
import numpy as np
import torch

torch.manual_seed(42)
x = torch.randn(1, 1, 257).numpy()*0.001
h1 = torch.randn(1, 1, 400).numpy()*0.001
h2 = torch.randn(1, 1, 400).numpy()*0.001

numpy_weights_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
baseline = NsNet2_npy(numpy_weights_path)
y = baseline(x, h1, h2)

mpq_config = np.ones(93)*32
quantized = Q_NsNet2_npy(numpy_weights_path, mpq_config)
yq = quantized(x, h1, h2)

ref_path = '/home/alessandro/Desktop/nas-nsnet2/examples/pesq/reference2.wav'
deg_path = '/home/alessandro/Desktop/nas-nsnet2/examples/pesq/degradated2.wav'
root_path = '/home/alessandro/Desktop/nas-nsnet2/'
build_path = '/home/alessandro/Desktop/build_nsnet2_nas/'
[pesq_metric, inference_metric, memory_metric] = compute_objective_function(mpq_config, numpy_weights_path, ref_path, deg_path, root_path, build_path)
print(f"normalized pesq: {pesq_metric}")
print(f"normalized inference time: {inference_metric}")
print(f"normalized memory footprint: {memory_metric}")