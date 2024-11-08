import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat

model_path = 'nsnet2_reimplemented.onnx'
model = onnx.load(model_path)

def get_calibration_data():
    calibration_data = []
    
    input1_cal = np.load('/media/alessandro/SecondDisk1/out_calibration/x.npy')
    input2_cal = np.load('/media/alessandro/SecondDisk1/out_calibration/h1.npy')
    input3_cal = np.load('/media/alessandro/SecondDisk1/out_calibration/h2.npy')
    num_calibration_samples = int(input3_cal.shape[0] / 400)
    num_calibration_samples = 10

    for i in range(num_calibration_samples):
        input1 = input1_cal[i*257:(i+1)*257]
        input1 = input1.reshape(1, 1, 257)
        input2 = input2_cal[i*400:(i+1)*400]
        input2 = input2.reshape(1, 1, 400)
        input3 = input3_cal[i*400:(i+1)*400]
        input3 = input3.reshape(1, 1, 400)
        calibration_data.append((input1, input2, input3))
    return calibration_data

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.data = get_calibration_data()
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_names = ['in_noisy', 'h1', 'h2']
            data = {
                input_names[0]: self.data[self.index][0],
                input_names[1]: self.data[self.index][1],
                input_names[2]: self.data[self.index][2]
            }
            self.index += 1
            return data
        return None

calibration_data_reader = MyCalibrationDataReader()
quantized_model_path = 'quantized_model_int8.onnx'

quant_format = QuantFormat.QOperator
quantized_model = quantize_static(
    model_input=model_path,
    model_output=quantized_model_path,
    calibration_data_reader=calibration_data_reader,        
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    quant_format=quant_format,
    #op_types_to_quantize=op_types_to_quantize,
)

print(f"new quantized model: {quantized_model_path}")
exit()
session = ort.InferenceSession(quantized_model_path)

sample_input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
sample_input2 = np.random.rand(1, 3, 224, 224).astype(np.float32)
sample_input3 = np.random.rand(1, 3, 224, 224).astype(np.float32)

input_names = [input.name for input in session.get_inputs()]
output = session.run(
    None,
    {
        input_names[0]: sample_input1,
        input_names[1]: sample_input2,
        input_names[2]: sample_input3
    }
)

print("output: ", output)