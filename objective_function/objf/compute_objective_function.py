from nsnet2.pytorch.nsnet2_q.nsnet2_quantized import Q_NsNet2_npy
from objective_function.pesq.compute_pesq import compute_pesq
from objective_function.inference_time.compute_inference_time import compute_inference_time
from objective_function.memory_footprint.compute_memory_footprint import compute_memory_footprint

def compute_objective_function(mpq_config, numpy_weights_path, ref_path, deg_path, root_path, build_path):
    quantized = Q_NsNet2_npy(numpy_weights_path, mpq_config)
    norm_pesq = compute_pesq(ref_path, deg_path, quantized)
    norm_inference_time = compute_inference_time(quantized.calib, root_path, build_path)
    norm_memory_footprint = compute_memory_footprint(mpq_config)
    return [1-norm_pesq, norm_inference_time, norm_memory_footprint]
