import numpy as np
import torch
from nsnet2.pytorch.nsnet2_q.calibration_param import init_calibration

class Q_NsNet2_npy(torch.nn.Module):
    def __init__(self, numpy_weights_path, mpq_config):
        super(Q_NsNet2_npy, self).__init__()

        self.calib = init_calibration(mpq_config)
        print(self.calib['fc1Add'].S())
        print(self.calib['fc1Add'].Z())

        # onnxMatMul_166
        self.onnxMatMul_166 = np.load(numpy_weights_path + 'onnx__MatMul_166.npy').transpose()
        self.onnxMatMul_166_q = self._quantize_tensor(self.onnxMatMul_166, 'onnxMatMul_166')

        # fc1bias
        self.fc1bias = np.load(numpy_weights_path + 'fc1_bias.npy')
        self.fc1bias_q = self._quantize_tensor(self.fc1bias, 'fc1bias')

        # Wiz_1, Wir_1, Win_1
        self.onnxGRU_184 = np.load(numpy_weights_path + 'onnx__GRU_184.npy')
        self.Wiz_1 = self.onnxGRU_184[:,:400,:]
        self.Wir_1 = self.onnxGRU_184[:,400:800,:]
        self.Win_1 = self.onnxGRU_184[:,800:,:]

        self.Wiz_1_q = self._quantize_tensor(self.Wiz_1, 'Wiz_1')
        self.Wir_1_q = self._quantize_tensor(self.Wir_1, 'Wir_1')
        self.Win_1_q = self._quantize_tensor(self.Win_1, 'Win_1')

        self.onnxGRU_185 = np.load(numpy_weights_path + 'onnx__GRU_185.npy')
        self.Whz_1 = self.onnxGRU_185[:,:400,:]
        self.Whr_1 = self.onnxGRU_185[:,400:800,:]
        self.Whn_1 = self.onnxGRU_185[:,800:,:]

        self.Whz_1_q = self._quantize_tensor(self.Whz_1, 'Whz_1')
        self.Whr_1_q = self._quantize_tensor(self.Whr_1, 'Whr_1')
        self.Whn_1_q = self._quantize_tensor(self.Whn_1, 'Whn_1')

        # biz_1, bir_1, bin_1, bhz_1, bhr_1, bhn_1
        self.onnxGRU_186 = np.load(numpy_weights_path + 'onnx__GRU_186.npy')
        self.biz_1 = self.onnxGRU_186[:,:400]
        self.bir_1 = self.onnxGRU_186[:,400:800]
        self.bin_1 = self.onnxGRU_186[:,800:1200]
        self.bhz_1 = self.onnxGRU_186[:,1200:1600]
        self.bhr_1 = self.onnxGRU_186[:,1600:2000]
        self.bhn_1 = self.onnxGRU_186[:,2000:]

        self.biz_1_q = self._quantize_tensor(self.biz_1, 'biz_1')
        self.bir_1_q = self._quantize_tensor(self.bir_1, 'bir_1')
        self.bin_1_q = self._quantize_tensor(self.bin_1, 'bin_1')
        self.bhz_1_q = self._quantize_tensor(self.bhz_1, 'bhz_1')
        self.bhr_1_q = self._quantize_tensor(self.bhr_1, 'bhr_1')
        self.bhn_1_q = self._quantize_tensor(self.bhn_1, 'bhn_1')

        self.onnxGRU_204 = np.load(numpy_weights_path + 'onnx__GRU_204.npy')
        self.Wiz_2 = self.onnxGRU_204[:,:400,:]
        self.Wir_2 = self.onnxGRU_204[:,400:800,:]
        self.Win_2 = self.onnxGRU_204[:,800:,:]

        self.Wiz_2_q = self._quantize_tensor(self.Wiz_2, 'Wiz_2')
        self.Wir_2_q = self._quantize_tensor(self.Wir_2, 'Wir_2')
        self.Win_2_q = self._quantize_tensor(self.Win_2, 'Win_2')

        self.onnxGRU_205 = np.load(numpy_weights_path + 'onnx__GRU_205.npy')
        self.Whz_2 = self.onnxGRU_205[:,:400,:]
        self.Whr_2 = self.onnxGRU_205[:,400:800,:]
        self.Whn_2 = self.onnxGRU_205[:,800:,:]

        self.Whz_2_q = self._quantize_tensor(self.Whz_2, 'Whz_2')
        self.Whr_2_q = self._quantize_tensor(self.Whr_2, 'Whr_2')
        self.Whn_2_q = self._quantize_tensor(self.Whn_2, 'Whn_2')

        self.onnxGRU_206 = np.load(numpy_weights_path + 'onnx__GRU_206.npy')
        self.biz_2 = self.onnxGRU_206[:,:400]
        self.bir_2 = self.onnxGRU_206[:,400:800]
        self.bin_2 = self.onnxGRU_206[:,800:1200]
        self.bhz_2 = self.onnxGRU_206[:,1200:1600]
        self.bhr_2 = self.onnxGRU_206[:,1600:2000]
        self.bhn_2 = self.onnxGRU_206[:,2000:]

        self.biz_2_q = self._quantize_tensor(self.biz_2, 'biz_2')
        self.bir_2_q = self._quantize_tensor(self.bir_2, 'bir_2')
        self.bin_2_q = self._quantize_tensor(self.bin_2, 'bin_2')
        self.bhz_2_q = self._quantize_tensor(self.bhz_2, 'bhz_2')
        self.bhr_2_q = self._quantize_tensor(self.bhr_2, 'bhr_2')
        self.bhn_2_q = self._quantize_tensor(self.bhn_2, 'bhn_2')

        self.onnxMatMul_207 = np.load(numpy_weights_path + 'onnx__MatMul_207.npy').transpose()
        self.onnxMatMul_207_q = self._quantize_tensor(self.onnxMatMul_207, 'onnxMatMul_207')
        
        self.fc2bias = np.load(numpy_weights_path + 'fc2_bias.npy')
        self.fc2bias_q = self._quantize_tensor(self.fc2bias, 'fc2bias')

        self.onnxMatMul_208 = np.load(numpy_weights_path + 'onnx__MatMul_208.npy').transpose()
        self.onnxMatMul_208_q = self._quantize_tensor(self.onnxMatMul_208, 'onnxMatMul_208')

        self.fc3bias = np.load(numpy_weights_path + 'fc3_bias.npy')
        self.fc3bias_q = self._quantize_tensor(self.fc3bias, 'fc3bias')

        self.onnxMatMul_209 = np.load(numpy_weights_path + 'onnx__MatMul_209.npy').transpose()
        self.onnxMatMul_209_q = self._quantize_tensor(self.onnxMatMul_209, 'onnxMatMul_209')

        self.fc4bias = np.load(numpy_weights_path + 'fc4_bias.npy')
        self.fc4bias_q = self._quantize_tensor(self.fc4bias, 'fc4bias')

    def forward(self, x, h1, h2):
        # process x
        x = x.squeeze()
        x_q = self._quantize_tensor(x, 'x')

        # h1
        h1 = h1.squeeze()
        h1_q = self._quantize_tensor(h1, 'h1')

        # h2
        h2 = h2.squeeze()
        h2_q = self._quantize_tensor(h2, 'h2')

        # fc1MatMul_q
        fc1MatMul_q = self._quantize_matmul(self.onnxMatMul_166_q, x_q, 'onnxMatMul_166', 'x', 'fc1MatMul')
        fc1MatMul = np.matmul(self.onnxMatMul_166, x)

        # fc1Add_q
        fc1Add_q = self._quantize_add(fc1MatMul_q, self.fc1bias_q, 'fc1MatMul', 'fc1bias', 'fc1Add')
        fc1Add = np.add(fc1MatMul, self.fc1bias)
        print(fc1Add_q.flatten()[0:10])
        print(np.sum(fc1Add_q))
        
        # gru1_a_
        gru1_a__q = self._quantize_matmul(self.Wir_1_q, fc1Add_q, 'Wir_1', 'fc1Add', 'gru1_a_')
        gru1_a_ = np.matmul(self.Wir_1, fc1Add)
        
        # gru1_a
        gru1_a_q = self._quantize_add(gru1_a__q, self.bir_1_q, 'gru1_a_', 'bir_1', 'gru1_a')
        gru1_a = np.add(gru1_a_, self.bir_1)

        # gru1_b_
        gru1_b__q = self._quantize_matmul(self.Whr_1_q, h1_q, 'Whr_1', 'h1', 'gru1_b_')
        gru1_b_ = np.matmul(self.Whr_1, h1)

        # gru1_b
        gru1_b_q = self._quantize_add(gru1_b__q, self.bhr_1_q, 'gru1_b_', 'bhr_1', 'gru1_b')
        gru1_b = np.add(gru1_b_, self.bhr_1)

        # gru1_c_
        gru1_c__q = self._quantize_matmul(self.Wiz_1_q, fc1Add_q, 'Wiz_1', 'fc1Add', 'gru1_c_')
        gru1_c_ = np.matmul(self.Wiz_1, fc1Add)

        # gru1_c
        gru1_c_q = self._quantize_add(gru1_c__q, self.biz_1_q, 'gru1_c_', 'biz_1', 'gru1_c')
        gru1_c = np.add(gru1_c_, self.biz_1)
        
        # gru1_d_
        gru1_d__q = self._quantize_matmul(self.Whz_1_q, h1_q, 'Whz_1', 'h1', 'gru1_d_')
        gru1_d_ = np.matmul(self.Whz_1, h1)
        
        # gru1_d
        gru1_d_q = self._quantize_add(gru1_d__q, self.bhz_1_q, 'gru1_d_', 'bhz_1', 'gru1_d')
        gru1_d = np.add(gru1_d_, self.bhz_1)

        # gru1_e_
        gru1_e__q = self._quantize_matmul(self.Win_1_q, fc1Add_q, 'Win_1', 'fc1Add', 'gru1_e_')
        gru1_e_ = np.matmul(self.Win_1, fc1Add) 

        # gru1_e
        gru1_e_q = self._quantize_add(gru1_e__q, self.bin_1_q, 'gru1_e_', 'bin_1', 'gru1_e')
        gru1_e = np.add(gru1_e_, self.bin_1)
        
        # gru1_f_
        gru1_f__q = self._quantize_matmul(self.Whn_1_q, h1_q, 'Whn_1', 'h1', 'gru1_f_')
        gru1_f_ = np.matmul(self.Whn_1, h1)
    
        # gru1_f
        gru1_f_q = self._quantize_add(gru1_f__q, self.bhn_1_q, 'gru1_f_', 'bhn_1', 'gru1_f')
        gru1_f = np.add(gru1_f_, self.bhn_1)

        # gru1_r_
        gru1_r__q = self._quantize_add(gru1_a_q, gru1_b_q, 'gru1_a', 'gru1_b', 'gru1_r_')
        gru1_r_ = np.add(gru1_a, gru1_b)

        # gru1_r
        temp_x = self._dequantize(gru1_r__q, self.calib['gru1_r_'].S(), self.calib['gru1_r_'].Z())
        temp_y = 1 / (1 + np.exp(-temp_x))
        gru1_r_q = self._quantize(temp_y, self.calib['gru1_r'].S(), self.calib['gru1_r'].Z(), self.calib['gru1_r'].bitwidth)
        gru1_r = 1 / (1 + np.exp(-gru1_r_))

        # gru1_z_
        gru1_z__q = self._quantize_add(gru1_c_q, gru1_d_q, 'gru1_c', 'gru1_d', 'gru1_z_')
        gru1_z_ = np.add(gru1_c, gru1_d)
        
        # gru1_z
        temp_x = self._dequantize(gru1_z__q, self.calib['gru1_z_'].S(), self.calib['gru1_z_'].Z())
        temp_y = 1 / (1 + np.exp(-temp_x))
        gru1_z_q = self._quantize(temp_y, self.calib['gru1_z'].S(), self.calib['gru1_z'].Z(), self.calib['gru1_z'].bitwidth)
        gru1_z = 1 / (1 + np.exp(-gru1_z_))

        # gru1_n1
        gru1_n1 = gru1_r * gru1_f
        gru1_n1_q = self._quantize_mul(gru1_r_q, gru1_f_q, 'gru1_r', 'gru1_f', 'gru1_n1')

        # gru1_n2
        gru1_n2_q = self._quantize_add(gru1_e_q, gru1_n1_q, 'gru1_e', 'gru1_n1', 'gru1_n2')
        gru1_n2 = np.add(gru1_e, gru1_n1)
        
        # gru1_n
        temp_x = self._dequantize(gru1_n2_q, self.calib['gru1_n2'].S(), self.calib['gru1_n2'].Z())
        temp_y = np.tanh(temp_x)
        gru1_n_q = self._quantize(temp_y, self.calib['gru1_n'].S(), self.calib['gru1_n'].Z(), self.calib['gru1_n'].bitwidth)
        gru1_n = np.tanh(gru1_n2)

        # gru1_hn1
        gru1_hn1_q = self._quantize_one_minus_x(gru1_z_q, 'gru1_z', 'gru1_hn1')
        gru1_hn1 = 1 - gru1_z

        # gru_hn2
        gru1_hn2_q = self._quantize_mul(gru1_hn1_q, gru1_n_q, 'gru1_hn1', 'gru1_n', 'gru1_hn2')
        gru1_hn2 = gru1_hn1 * gru1_n

        # gru_hn3
        gru1_hn3_q = self._quantize_mul(gru1_z_q, h1_q, 'gru1_z', 'h1', 'gru1_hn3')
        gru1_hn3 = gru1_z * h1

        # rnn1GRU
        rnn1GRU_q = self._quantize_add(gru1_hn2_q, gru1_hn3_q, 'gru1_hn2', 'gru1_hn3', 'rnn1GRU')
        rnn1GRU = np.add(gru1_hn2, gru1_hn3)
        
        # gru 2
        rnn1GRU = rnn1GRU.squeeze()
        rnn1GRU_q = rnn1GRU_q.squeeze()

        # gru2_a_   
        gru2_a__q = self._quantize_matmul(self.Wir_2_q, rnn1GRU_q, 'Wir_2', 'rnn1GRU', 'gru2_a_')
        gru2_a_ = np.matmul(self.Wir_2, rnn1GRU)
        
        # gru2_a
        gru2_a_q = self._quantize_add(gru2_a__q, self.bir_2_q, 'gru2_a_', 'bir_2', 'gru2_a')
        gru2_a = np.add(gru2_a_, self.bir_2)
        
        # gru2_b_
        gru2_b__q = self._quantize_matmul(self.Whr_2_q, h2_q, 'Whr_2', 'h2', 'gru2_b_')
        gru2_b_ = np.matmul(self.Whr_2, h2)
        
        # gru2_b
        gru2_b_q = self._quantize_add(gru2_b__q, self.bhr_2_q, 'gru2_b_', 'bhr_2', 'gru2_b')
        gru2_b = np.add(gru2_b_, self.bhr_2)
        
        # gru2_c_
        gru2_c__q = self._quantize_matmul(self.Wiz_2_q, rnn1GRU_q, 'Wiz_2', 'rnn1GRU', 'gru2_c_')
        gru2_c_ = np.matmul(self.Wiz_2, rnn1GRU)

        # gru2_c
        gru2_c_q = self._quantize_add(gru2_c__q, self.biz_2_q, 'gru2_c_', 'biz_2', 'gru2_c')
        gru2_c = np.add(gru2_c_, self.biz_2)
        
        # gru2_d_
        gru2_d__q = self._quantize_matmul(self.Whz_2_q, h2_q, 'Whz_2', 'h2', 'gru2_d_')
        gru2_d_ = np.matmul(self.Whz_2, h2)
        
        # gru2_d
        gru2_d_q = self._quantize_add(gru2_d__q, self.bhz_2_q, 'gru2_d_', 'bhz_2', 'gru2_d')
        gru2_d = np.add(gru2_d_, self.bhz_2)

        # gru_e_
        gru2_e__q = self._quantize_matmul(self.Win_2_q, rnn1GRU_q, 'Win_2', 'rnn1GRU', 'gru2_e_')
        gru2_e_ = np.matmul(self.Win_2, rnn1GRU)
        
        # gru2_e
        gru2_e_q = self._quantize_add(gru2_e__q, self.bin_2_q, 'gru2_e_', 'bin_2', 'gru2_e')
        gru2_e = np.add(gru2_e_, self.bin_2)

        # gru2_f_
        gru2_f__q = self._quantize_matmul(self.Whn_2_q, h2_q, 'Whn_2', 'h2', 'gru2_f_')
        gru2_f_ = np.matmul(self.Whn_2, h2)
        
        # gru2_f
        gru2_f_q = self._quantize_add(gru2_f__q, self.bhn_2_q, 'gru2_f_', 'bhn_2', 'gru2_f')
        gru2_f = np.add(gru2_f_, self.bhn_2)

        # gru2_r_
        gru2_r__q = self._quantize_add(gru2_a_q, gru2_b_q, 'gru2_a', 'gru2_b', 'gru2_r_')
        gru2_r_ = np.add(gru2_a, gru2_b)

        # gru2_r
        temp_x = self._dequantize(gru2_r__q, self.calib['gru2_r_'].S(), self.calib['gru2_r_'].Z())
        temp_y = 1 / (1 + np.exp(-temp_x))
        gru2_r_q = self._quantize(temp_y, self.calib['gru2_r'].S(), self.calib['gru2_r'].Z(), self.calib['gru2_r'].bitwidth)
        gru2_r = 1 / (1 + np.exp(-gru2_r_))

        # gru2_z_
        gru2_z__q = self._quantize_add(gru2_c_q, gru2_d_q, 'gru2_c', 'gru2_d', 'gru2_z_')
        gru2_z_ = np.add(gru2_c, gru2_d)
        
        # gru2_z
        temp_x = self._dequantize(gru2_z__q, self.calib['gru2_z_'].S(), self.calib['gru2_z_'].Z())
        temp_y = 1 / (1 + np.exp(-temp_x))
        gru2_z_q = self._quantize(temp_y, self.calib['gru2_z'].S(), self.calib['gru2_z'].Z(), self.calib['gru2_z'].bitwidth)
        gru2_z = 1 / (1 + np.exp(-gru2_z_))

        # gru2_n1
        gru2_n1_q = self._quantize_mul(gru2_r_q, gru2_f_q, 'gru2_r', 'gru2_f', 'gru2_n1')
        gru2_n1 = gru2_r * gru2_f
        
        # gru2_n2
        gru2_n2_q = self._quantize_add(gru2_e_q, gru2_n1_q, 'gru2_e', 'gru2_n1', 'gru2_n2')
        gru2_n2 = np.add(gru2_e, gru2_n1)

        # gru2_n
        temp_x = self._dequantize(gru2_n2_q, self.calib['gru2_n2'].S(), self.calib['gru2_n2'].Z())
        temp_y = np.tanh(temp_x)
        gru2_n_q = self._quantize(temp_y, self.calib['gru2_n'].S(), self.calib['gru2_n'].Z(), self.calib['gru2_n'].bitwidth)
        gru2_n = np.tanh(gru2_n2)

        # gru2_hn1
        gru2_hn1_q = self._quantize_one_minus_x(gru2_z_q, 'gru2_z', 'gru2_hn1')
        gru2_hn1 = 1 - gru2_z
        
        # gru2_hn2
        gru2_hn2_q = self._quantize_mul(gru2_hn1_q, gru2_n_q, 'gru2_hn1', 'gru2_n', 'gru2_hn2')
        gru2_hn2 = gru2_hn1 * gru2_n

        # gru2_hn3
        gru2_hn3_q = self._quantize_mul(gru2_z_q, h2_q, 'gru2_z', 'h2', 'gru2_hn3')
        gru2_hn3 = gru2_z * h2

        # rnn2GRU
        rnn2GRU_q = self._quantize_add(gru2_hn2_q, gru2_hn3_q, 'gru2_hn2', 'gru2_hn3', 'rnn2GRU')
        rnn2GRU = np.add(gru2_hn2, gru2_hn3)

        rnn2GRU = rnn2GRU.squeeze()
        rnn2GRU_q = rnn2GRU_q.squeeze()

        # fc2MatMul
        fc2MatMul_q = self._quantize_matmul(self.onnxMatMul_207_q, rnn2GRU_q, 'onnxMatMul_207', 'rnn2GRU', 'fc2MatMul')
        fc2MatMul = np.matmul(self.onnxMatMul_207, rnn2GRU)
        
        # fc2Add
        fc2Add_q = self._quantize_add(fc2MatMul_q, self.fc2bias_q, 'fc2MatMul', 'fc2bias', 'fc2Add')
        fc2Add = np.add(fc2MatMul, self.fc2bias)
        
        # relu
        relu_q = self._quantize_relu(fc2Add_q, 'fc2Add', 'relu')
        relu = np.maximum(0, fc2Add)

        # fc3MatMul
        fc3MatMul_q = self._quantize_matmul(self.onnxMatMul_208_q, relu_q, 'onnxMatMul_208', 'relu', 'fc3MatMul')
        fc3MatMul = np.matmul(self.onnxMatMul_208, relu)

        # fc3Add
        fc3Add_q = self._quantize_add(fc3MatMul_q, self.fc3bias_q, 'fc3MatMul', 'fc3bias', 'fc3Add')
        fc3Add = np.add(fc3MatMul, self.fc3bias)
        
        # relu_1
        relu_1_q = self._quantize_relu(fc3Add_q, 'fc3Add', 'relu_1')
        relu_1 = np.maximum(0, fc3Add)

        # fc4MatMul
        fc4MatMul_q = self._quantize_matmul(self.onnxMatMul_209_q, relu_1_q, 'onnxMatMul_209', 'relu_1', 'fc4MatMul')
        fc4MatMul = np.matmul(self.onnxMatMul_209, relu_1)
        
        # fc4Add
        fc4Add_q = self._quantize_add(fc4MatMul_q, self.fc4bias_q, 'fc4MatMul', 'fc4bias', 'fc4Add')
        fc4Add = np.add(fc4MatMul, self.fc4bias)

        # sigmoid
        sigmoid_q = self._quantize_sigmoid(fc4Add_q, 'fc4Add', 'sigmoid')
        sigmoid = 1 / (1 + np.exp(-fc4Add))
        #self._compare(sigmoid, sigmoid_q, self.calib['sigmoid'])

        return sigmoid_q
    
    def _quantize(self, tensor_fp32, S, z, n_bits):
        q = np.floor(tensor_fp32 / S) + z
        q = q % (2**n_bits)
        return q
    
    def _dequantize(self, tensor_i8, S, z):
        return (tensor_i8 - z) * S

    def _compare(self, tensor_f32, tensor_int, calib):
        print(np.mean(np.abs(tensor_f32 - self._dequantize(tensor_int, calib.S(), calib.Z()))))
    
    def _quantize_tensor(self, tensor_f32, c_key):
        c = self.calib[c_key]
        return self._quantize(tensor_f32, c.S(), c.Z(), c.bitwidth)
    
    def _quantize_matmul(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        S = (ca.S()*cb.S() / cy.S())
        return np.round(
            S * np.matmul(A-ca.Z(), B-cb.Z()) + cy.Z()
        )
    
    def _quantize_add(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        return np.round(
            (ca.S() / cy.S()) * (A - ca.Z()) + (cb.S() / cy.S()) * (B - cb.Z()) + cy.Z()
        )
    
    def _quantize_mul(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        return (ca.S() * cb.S() / cy.S()) * (A - ca.Z()) * (B - cb.Z()) + cy.Z()
    
    def _quantize_one_minus_x(self, X, cx_key, cy_key):
        cx = self.calib[cx_key]
        cy = self.calib[cy_key]
        one_q = self._quantize(np.ones(X.shape), cx.S(), cx.Z(), cx.bitwidth)
        return (cx.S() / cy.S()) * (one_q - X) + cy.Z()
    
    def _quantize_relu(self, X_q, cx_key, cy_key):
        cx = self.calib[cx_key]
        cy = self.calib[cy_key]
        X = self._dequantize(X_q, cx.S(), cx.Z())
        Y = np.maximum(0, X)
        return self._quantize(Y, cy.S(), cy.Z(), cy.bitwidth)
    
    def _quantize_sigmoid(self, X_q, cx_key, cy_key):
        cx = self.calib[cx_key]
        cy = self.calib[cy_key]
        X = self._dequantize(X_q, cx.S(), cx.Z())
        Y = 1 / (1 + np.exp(-X))
        return self._quantize(Y, cy.S(), cy.Z(), cy.bitwidth)