import numpy as np

model_layers_size = np.array([
    257, 257*400, 400, 
    400, 400, 400*400, 
    400, 400, 400, 
    400, 400, 400, 
    400*400, 400*400, 400*400,
    400*400, 400*400, 400,
    400, 400, 400,
    400, 400*400, 400*400,
    400*400, 400*400, 400*400,
    400*400, 400, 400,
    400, 400, 400,
    400, 400*600, 600,
    600*600, 600, 600*257,
    257, 400, 400,
    400, 400, 400, 
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    400, 400, 400,
    600, 600, 600,
    600, 600, 600,
    257, 257, 257
], dtype=np.int32)

def compute_memory_footprint(mpq_config):
    n_bits_model = _num_bits_model(mpq_config)
    max_n_bits_model = _max_num_bits_model()
    min_n_bits_model = _min_num_bits_model()
    normalized = _normalize_metric(n_bits_model, min_n_bits_model, max_n_bits_model)
    return normalized

def _num_bits_model(mpq_config):
    return np.sum(mpq_config * model_layers_size)

def _max_num_bits_model():
    max_bitwidth = 32
    return np.sum(model_layers_size * max_bitwidth)

def _min_num_bits_model():
    min_bitwidth = 8
    return np.sum(model_layers_size * min_bitwidth)

def _normalize_metric(bitwidth, min_bitwidth, max_bitwidth):
    return (bitwidth - min_bitwidth) / (max_bitwidth - min_bitwidth)

