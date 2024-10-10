
import os
import numpy as np

def extract_metrics(out_calibration_path):
    # read audio files
    all_files = os.listdir(out_calibration_path)
    only_files = [f for f in all_files if os.path.isfile(os.path.join(out_calibration_path, f))]

    result = "####################################################################\n"
    result += "# This code is generated automatically from run_calibration.py\n"
    result += "####################################################################\n\n"
    for file in only_files:
        tensor = np.load(out_calibration_path + file)
        name = file.split('.')[0]
        result += f"# {name}\n"
        result += f"{name}_min = {np.min(tensor)}\n"
        result += f"{name}_max = {np.max(tensor)}\n"
        result += "\n"
    
    with open(out_calibration_path + 'calibration_output.py', 'w') as file:
        file.write(result)
