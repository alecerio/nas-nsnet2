import onnxruntime as ort
import numpy as np


onnx_model_path = 'nsnet2.onnx'
session = ort.InferenceSession(onnx_model_path)

input_names = [input.name for input in session.get_inputs()]
output_name = session.get_outputs()[0].name

input_tensor = np.ones((1, 1, 257), dtype=np.float32) 
h1_tensor = np.ones((1, 1, 400), dtype=np.float32)
h2_tensor = np.ones((1, 1, 400), dtype=np.float32)

input_dict = {
    input_names[0]: input_tensor,
    input_names[1]: h1_tensor,     
    input_names[2]: h2_tensor      
}

output = session.run([output_name], input_dict)

print(f"Output: {output}")
