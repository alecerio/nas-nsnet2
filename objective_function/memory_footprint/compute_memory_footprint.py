import numpy as np

def compute_memory_footprint(mpq_config):
    n_bits_model = _num_bits_model(mpq_config)
    max_n_bits_model = _max_num_bits_model(mpq_config)
    min_n_bits_model = _min_num_bits_model(mpq_config)
    normalized = _normalize_metric(n_bits_model, min_n_bits_model, max_n_bits_model)
    return normalized

def _num_bits_model(mpq_config):
    return np.sum(mpq_config)

def _max_num_bits_model(mpq_config):
    n_params = len(mpq_config)
    max_bitwidth = 32
    return n_params * max_bitwidth

def _min_num_bits_model(mpq_config):
    n_params = len(mpq_config)
    min_bitwidth = 8
    return n_params * min_bitwidth

def _normalize_metric(bitwidth, min_bitwidth, max_bitwidth):
    return (bitwidth - min_bitwidth) / (max_bitwidth - min_bitwidth)

