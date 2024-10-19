# see https://github.com/vBaiCai/python-pesq

import soundfile as sf
from pypesq import pesq
import torch
import onnxruntime as ort
import numpy as np

ref, sr = sf.read('reference2.wav')
deg, sr = sf.read('degradated2.wav')

waveform = torch.tensor(deg, dtype=torch.float32)
window_size = 512
overlap = 0.5
hop_length = int(window_size * (1 - overlap))
stft_result = torch.stft(waveform, n_fft=window_size, hop_length=hop_length, win_length=window_size, return_complex=True)

session = ort.InferenceSession('nsnet2.onnx')
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]

output = np.zeros(stft_result.shape).astype(np.cfloat)
output = torch.from_numpy(output)
h1 = np.zeros([1, 1, 400]).astype(np.float32)
h2 = np.zeros([1, 1, 400]).astype(np.float32)
num_frames = stft_result.shape[1]
for i in range(0, num_frames):
    input_data = stft_result[:,i]
    input_data_mod = torch.log(input_data.abs() ** 2 + 10e-9)
    input_data_mod = np.expand_dims(np.expand_dims(input_data_mod, axis=0), axis=0)
    inputs = {
        input_names[0]: input_data_mod,
        input_names[1]: h1,
        input_names[2]: h2
    }
    outputs = session.run(output_names, inputs)
    output[:,i] = (torch.Tensor(outputs[0]) * input_data).squeeze()
    h1 = outputs[1]
    h2 = outputs[2]

#output = torch.from_numpy(output)
reconstructed = torch.istft(output, n_fft=window_size, hop_length=hop_length, win_length=window_size, return_complex=False)

print(deg.shape)
print(reconstructed.shape)
sf.write('rec.wav', reconstructed, sr)

score = pesq(ref, reconstructed, sr)
print(score)