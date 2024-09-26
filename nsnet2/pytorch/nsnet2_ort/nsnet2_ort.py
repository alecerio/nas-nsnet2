import onnxruntime as ort
import numpy as np

class NsNetORT():
    def __init__(self, onnx_model_path):
        super(NsNetORT, self).__init__()
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
    
    def run(self, input_tensor, h1_tensor, h2_tensor):
        input_dict = {
            self.input_names[0]: input_tensor,
            self.input_names[1]: h1_tensor,     
            self.input_names[2]: h2_tensor      
        }
        output = self.session.run([self.output_name], input_dict)
        return output





#input_tensor = np.ones((1, 1, 257), dtype=np.float32) 
#h1_tensor = np.ones((1, 1, 400), dtype=np.float32)*2
#h2_tensor = np.ones((1, 1, 400), dtype=np.float32)
#nsnet_ort = NsNetORT()
#output = nsnet_ort.run(input_tensor, h1_tensor, h2_tensor)
#print(f"Output: {output}")
