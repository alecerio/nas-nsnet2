import soundfile as sf
from pypesq import pesq
import torch
import onnxruntime as ort
import numpy as np

def compute_pesq(ref_wav_path, deg_wav_path, nsnet2):
    ref, sr = sf.read(ref_wav_path)
    deg, sr = sf.read(deg_wav_path)

    waveform = torch.tensor(deg, dtype=torch.float32)
    window_size = 512
    overlap = 0.5
    hop_length = int(window_size * (1 - overlap))
    stft_result = torch.stft(waveform, n_fft=window_size, hop_length=hop_length, win_length=window_size, return_complex=True)

    output = np.zeros(stft_result.shape).astype(np.cfloat)
    output = torch.from_numpy(output)
    h1 = np.zeros([1, 1, 400]).astype(np.float32)
    h2 = np.zeros([1, 1, 400]).astype(np.float32)
    num_frames = stft_result.shape[1]
    for i in range(0, num_frames):
        input_data = stft_result[:,i]
        input_data_mod = torch.log(input_data.abs() ** 2 + 10e-9)
        input_data_mod = np.expand_dims(np.expand_dims(input_data_mod, axis=0), axis=0)
        [out, h1n, h2n] = nsnet2(input_data_mod, h1, h2)
        output[:,i] = (torch.Tensor(out) * input_data).squeeze()
        h1 = h1n
        h2 = h2n

    reconstructed = torch.istft(output, n_fft=window_size, hop_length=hop_length, win_length=window_size, return_complex=False)
    score = pesq(ref, reconstructed, sr)
    normalized_score = _normalize_pesq(score)
    return normalized_score

def _normalize_pesq(pesq_score):
    min_pesq = 0.5
    max_pesq = 4.5
    normalized_pesq = (pesq_score - min_pesq) / (max_pesq - min_pesq)
    return normalized_pesq