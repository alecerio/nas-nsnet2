import numpy as np
import torch

class CalibrationParam():
    def __init__(self, bitwidth, signed, minimum, maximum) -> None:
        self.bitwidth = bitwidth
        self.signed = signed
        self.minimum = minimum
        self.maximum = maximum
    
    def S(self):
        if self.signed:
            return (self.maximum - self.minimum) / (2 ** (self.bitwidth-1) - 1)
        else:
            return (self.maximum - self.minimum) / (2 ** (self.bitwidth) - 1)
    
    def Z(self):
        if self.signed:
            return 0
        else:
            return round(-(self.minimum / self.S()))


class Q_NsNet2_npy(torch.nn.Module):
    def __init__(self):
        super(Q_NsNet2_npy, self).__init__()

        self.calib = {}
        self.calib['x'] = CalibrationParam(8, False, -0.0025095, 0.0022181)
        self.calib['onnxMatMul_166'] = CalibrationParam(8, False, -0.22075387835502625, 0.208940327167511)
        self.calib['fc1MatMul'] = CalibrationParam(8, False, -0.00291599917, 0.0017367251)
        self.calib['fc1bias'] = CalibrationParam(8, False, -0.48688140511512756, 0.5176185369491577)
        self.calib['fc1Add'] = CalibrationParam(8, False, -0.48778465390205383, 0.5181604027748108)
        self.calib['Wir_1'] = CalibrationParam(8, False, -0.34401071071624756, 0.29191476106643677)
        self.calib['gru1_a_'] = CalibrationParam(8, False, -0.6389939785003662, 0.7715625762939453)

        # weights

        self.onnxMatMul_166 = np.load('onnx__MatMul_166.npy').transpose()
        self.fc1bias = np.load('fc1_bias.npy')

        self.onnxGRU_184 = np.load('onnx__GRU_184.npy')
        self.Wiz_1 = self.onnxGRU_184[:,:400,:]
        self.Wir_1 = self.onnxGRU_184[:,400:800,:]
        self.Win_1 = self.onnxGRU_184[:,800:,:]

        self.onnxGRU_185 = np.load('onnx__GRU_185.npy')
        self.Whz_1 = self.onnxGRU_185[:,:400,:]
        self.Whr_1 = self.onnxGRU_185[:,400:800,:]
        self.Whn_1 = self.onnxGRU_185[:,800:,:]

        self.onnxGRU_186 = np.load('onnx__GRU_186.npy')
        self.biz_1 = self.onnxGRU_186[:,:400]
        self.bir_1 = self.onnxGRU_186[:,400:800]
        self.bin_1 = self.onnxGRU_186[:,800:1200]
        self.bhz_1 = self.onnxGRU_186[:,1200:1600]
        self.bhr_1 = self.onnxGRU_186[:,1600:2000]
        self.bhn_1 = self.onnxGRU_186[:,2000:]

        self.onnxGRU_204 = np.load('onnx__GRU_204.npy')
        self.Wiz_2 = self.onnxGRU_204[:,:400,:]
        self.Wir_2 = self.onnxGRU_204[:,400:800,:]
        self.Win_2 = self.onnxGRU_204[:,800:,:]

        self.onnxGRU_205 = np.load('onnx__GRU_205.npy')
        self.Whz_2 = self.onnxGRU_205[:,:400,:]
        self.Whr_2 = self.onnxGRU_205[:,400:800,:]
        self.Whn_2 = self.onnxGRU_205[:,800:,:]

        self.onnxGRU_206 = np.load('onnx__GRU_206.npy')
        self.biz_2 = self.onnxGRU_206[:,:400]
        self.bir_2 = self.onnxGRU_206[:,400:800]
        self.bin_2 = self.onnxGRU_206[:,800:1200]
        self.bhz_2 = self.onnxGRU_206[:,1200:1600]
        self.bhr_2 = self.onnxGRU_206[:,1600:2000]
        self.bhn_2 = self.onnxGRU_206[:,2000:]

        self.onnxMatMul_207 = np.load('onnx__MatMul_207.npy').transpose()
        self.fc2bias = np.load('fc2_bias.npy')

        self.onnxMatMul_208 = np.load('onnx__MatMul_208.npy').transpose()
        self.fc3bias = np.load('fc3_bias.npy')

        self.onnxMatMul_209 = np.load('onnx__MatMul_209.npy').transpose()
        self.fc4bias = np.load('fc4_bias.npy')

    def forward(self, x, h1, h2):
        # process x
        x = x.squeeze()
        c = self.calib['x']
        x_q = self._quantize(x, c.S(), c.Z())

        h1 = h1.squeeze()
        h2 = h2.squeeze()

        # process fc1MatMul_q
        ca = self.calib['onnxMatMul_166']
        cb = self.calib['x']
        cy = self.calib['fc1MatMul']
        onnxMatMul_166_q = self._quantize(self.onnxMatMul_166, ca.S(), ca.Z())
        
        #fc1MatMul_q = np.round((ca.S()*cb.S() / cy.S()) * ( np.matmul(onnxMatMul_166_q, x_q)[:,None] -cb.Z()*onnxMatMul_166_q -ca.Z()*x_q + ca.Z()*cb.Z() ) + cy.Z())
        #fc1MatMul_q = np.round((ca.S()*cb.S() / cy.S()) * ( np.matmul(onnxMatMul_166_q, x_q)[:,None] -cb.Z()*onnxMatMul_166_q -ca.Z()*x_q + ca.Z()*cb.Z() ) + cy.Z())
        fc1MatMul_q = np.round(
            (ca.S()*cb.S() / cy.S()) * np.matmul(onnxMatMul_166_q - ca.Z(), x_q - cb.Z()) + cy.Z()
        )

        # to remove (float32)
        fc1MatMul = np.matmul(self.onnxMatMul_166, x)

        # fc1Add_q
        ca = self.calib['fc1MatMul']
        cb = self.calib['fc1bias']
        cy = self.calib['fc1Add']
        fc1bias_q = self._quantize(self.fc1bias, cb.S(), cb.Z())
        fc1Add_q = (ca.S() / cy.S()) * (fc1MatMul_q - ca.Z()) + (cb.S() / cy.S()) * (fc1bias_q - cb.Z()) + cy.Z()
        
        # to remove (float32)
        fc1Add = np.add(fc1MatMul, self.fc1bias)
        
        # gru1_a_
        ca = self.calib['Wir_1']
        cb = self.calib['fc1Add']
        cy = self.calib['gru1_a_']
        Wir_1_q = self._quantize(self.Wir_1, ca.S(), ca.Z())
        gru1_a__q = np.round(
            (ca.S()*cb.S() / cy.S()) * np.matmul(Wir_1_q - ca.Z(), fc1Add_q - cb.Z()) + cy.Z()
        )
        
        # to remove (float32)
        gru1_a_ = np.matmul(self.Wir_1, fc1Add)
        
        gru1_a = np.add(gru1_a_, self.bir_1)
        gru1_b_ = np.matmul(self.Whr_1, h1)
        gru1_b = np.add(gru1_b_, self.bhr_1)
        gru1_c_ = np.matmul(self.Wiz_1, fc1Add)
        gru1_c = np.add(gru1_c_, self.biz_1)
        gru1_d_ = np.matmul(self.Whz_1, h1)
        gru1_d = np.add(gru1_d_, self.bhz_1)
        gru1_e_ = np.matmul(self.Win_1, fc1Add)
        gru1_e = np.add(gru1_e_, self.bin_1)
        gru1_f_ = np.matmul(self.Whn_1, h1)
        gru1_f = np.add(gru1_f_, self.bhn_1)

        gru1_r_ = np.add(gru1_a, gru1_b)
        gru1_r = 1 / (1 + np.exp(-gru1_r_))

        gru1_z_ = np.add(gru1_c, gru1_d)
        gru1_z = 1 / (1 + np.exp(-gru1_z_))

        gru1_n1 = gru1_r * gru1_f
        gru1_n2 = np.add(gru1_e, gru1_n1)
        gru1_n = np.tanh(gru1_n2)

        gru1_hn1 = 1 - gru1_z
        gru1_hn2 = gru1_hn1 * gru1_n
        gru1_hn3 = gru1_z * h1
        rnn1GRU = np.add(gru1_hn2, gru1_hn3)

        # gru 2
        rnn1GRU = rnn1GRU.squeeze()
        gru2_a_ = np.matmul(self.Wir_2, rnn1GRU)
        gru2_a = np.add(gru2_a_, self.bir_2)
        gru2_b_ = np.matmul(self.Whr_2, h2)
        gru2_b = np.add(gru2_b_, self.bhr_2)
        gru2_c_ = np.matmul(self.Wiz_2, rnn1GRU)
        gru2_c = np.add(gru2_c_, self.biz_2)
        gru2_d_ = np.matmul(self.Whz_2, h2)
        gru2_d = np.add(gru2_d_, self.bhz_2)
        gru2_e_ = np.matmul(self.Win_2, rnn1GRU)
        gru2_e = np.add(gru2_e_, self.bin_2)
        gru2_f_ = np.matmul(self.Whn_2, h2)
        gru2_f = np.add(gru2_f_, self.bhn_2)

        gru2_r_ = np.add(gru2_a, gru2_b)
        gru2_r = 1 / (1 + np.exp(-gru2_r_))

        gru2_z_ = np.add(gru2_c, gru2_d)
        gru2_z = 1 / (1 + np.exp(-gru2_z_))

        gru2_n1 = gru2_r * gru2_f
        gru2_n2 = np.add(gru2_e, gru2_n1)
        gru2_n = np.tanh(gru2_n2)

        gru2_hn1 = 1 - gru2_z
        gru2_hn2 = gru2_hn1 * gru2_n
        gru2_hn3 = gru2_z * h2
        rnn2GRU = np.add(gru2_hn2, gru2_hn3)
        rnn2GRU = rnn2GRU.squeeze()

        # fully connected 2
        fc2MatMul = np.matmul(self.onnxMatMul_207, rnn2GRU)
        fc2Add = np.add(fc2MatMul, self.fc2bias)
        relu = np.maximum(0, fc2Add)

        # fully connected 3
        fc3MatMul = np.matmul(self.onnxMatMul_208, relu)
        fc3Add = np.add(fc3MatMul, self.fc3bias)
        relu_1 = np.maximum(0, fc3Add)

        # fully connected 4
        fc4MatMul = np.matmul(self.onnxMatMul_209, relu_1)
        fc4Add = np.add(fc4MatMul, self.fc4bias)
        sigmoid = 1 / (1 + np.exp(-fc4Add))

        return sigmoid
    
    def _quantize(self, tensor_fp32, S, z):
        return np.floor(tensor_fp32 / S) + z
    
    def _dequantize(self, tensor_i8, S, z):
        return (tensor_i8 - z) * S