
import os
from nsnet2.pytorch.calibration.nsnet2 import NsNet2_npy
import soundfile as sf
import torch
import numpy as np

def calibrate(dataset_path, weights_path, out_calibration_path, num_samples):
    # read audio files
    all_files = os.listdir(dataset_path)
    only_files = [f for f in all_files if os.path.isfile(os.path.join(dataset_path, f))]
    
    # initialize nsnet2
    nsnet2 = NsNet2_npy(weights_path)
    
    # calibrate
    i = 0
    for file in only_files[0:num_samples]:
        print(f"calibrate file {i} out of {num_samples}")
        _calibrate_on_file(dataset_path + file, nsnet2)
        i = i+1
    
    _save_calibration(nsnet2, out_calibration_path)

def _calibrate_on_file(path_audio_file, nsnet2):
    deg, sr = sf.read(path_audio_file)

    waveform = torch.tensor(deg, dtype=torch.float32)
    window_size = 512
    overlap = 0.5
    hop_length = int(window_size * (1 - overlap))
    stft_result = torch.stft(waveform, n_fft=window_size, hop_length=hop_length, win_length=window_size, return_complex=True)

    h1 = np.zeros([1, 1, 400]).astype(np.float32)
    h2 = np.zeros([1, 1, 400]).astype(np.float32)
    num_frames = stft_result.shape[1]
    for i in range(0, num_frames):
        input_data = stft_result[:,i]
        input_data_mod = torch.log(input_data.abs() ** 2 + 10e-9)
        input_data_mod = np.expand_dims(np.expand_dims(input_data_mod, axis=0), axis=0)
        outputs = nsnet2(input_data_mod, h1, h2)
        h1 = outputs[1]
        h2 = outputs[2]

def _save_calibration(nsnet2 : NsNet2_npy, out_calibration_path):
    for key in nsnet2.calib.calib_dict.keys():
        tensor = nsnet2.calib.calib_dict[key]
        np.save(out_calibration_path + key + '.npy', tensor)