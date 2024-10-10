from nsnet2.pytorch.calibration.calibration import calibrate
from nsnet2.pytorch.calibration.extract_metrics import extract_metrics

if __name__ == '__main__':
    dataset_path = '/media/alessandro/SecondDisk/dataset/gna_datasets-main/dns2020_6k/noisy/'
    weights_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
    out_calibration_path = '/media/alessandro/SecondDisk/out_calibration/'
    #calibrate(dataset_path, weights_path, out_calibration_path, 10)
    extract_metrics(out_calibration_path)

    