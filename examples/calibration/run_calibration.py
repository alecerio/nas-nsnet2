from nsnet2.pytorch.calibration.calibration import calibrate
from nsnet2.pytorch.calibration.extract_metrics import extract_metrics

if __name__ == '__main__':
    dataset_path = 'anonimized-for-double-blind-review'
    weights_path = 'anonimized-for-double-blind-review/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
    out_calibration_path = 'anonimized-for-double-blind-review/out_calibration/'
    calibrate(dataset_path, weights_path, out_calibration_path, 10)
    extract_metrics(out_calibration_path)

    