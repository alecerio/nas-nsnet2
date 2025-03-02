import torch
import numpy as np
from nsnet2.pytorch.nsnet2.nsnet2 import NsNet2_npy, NsNet2
from nsnet2.pytorch.nsnet2_ort.nsnet2_ort import NsNetORT
import torch.onnx

numpy_weights_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
onnx_model_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/nsnet2_ort/nsnet2.onnx'

result = True
for i in range(0, 100):
    x = torch.rand([1, 1, 257], dtype=torch.float32)
    h1 = torch.rand([1, 1, 400], dtype=torch.float32)
    h2 = torch.rand([1, 1, 400], dtype=torch.float32)
    
    nsnet = NsNet2_npy(numpy_weights_path)
    output = nsnet(x.numpy(), h1.numpy(), h2.numpy())
    output = output[0]

    nsnet_ort = NsNetORT(onnx_model_path)
    output_ort = nsnet_ort.run(x.numpy(), h1.numpy(), h2.numpy())
    output_ort_np = np.array(output_ort)
    output_ort_tensor = torch.tensor(output_ort_np)
    output_ort_tensor = output_ort_tensor.squeeze()

    diff = np.mean(np.abs(output-output_ort_np)) 
    if diff > 1e-06:
        test_res = False
    else:
        test_res = True
    print(f"{diff} -> {test_res}")

    if i == 0:
        nsnet_torch = NsNet2('/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/pytorch_weights/')
        torch.onnx.export(
            nsnet_torch,
            (x, h1, h2),
            "nsnet2_reimplemented.onnx",
            input_names=['x', 'h1', 'h2'],
            output_names=['y', 'h1n', 'h2n'],
            opset_version=11
        )

    result = result and test_res
print("The result is: " + str(result))