import torch
import numpy as np

class Calibration():
    def __init__(self) -> None:
        self.weights_calibrated = False
        keys = [
            'onnxMatMul_166', 'fc1bias', 'Wiz_1', 'Wir_1', 'Win_1', 'Whz_1', 'Whr_1',
            'Whn_1', 'biz_1', 'bir_1', 'bin_1', 'bhz_1', 'bhr_1', 'bhn_1', 'Wiz_2',
            'Wir_2', 'Win_2', 'Whz_2', 'Whr_2', 'Whn_2', 'biz_2', 'bir_2', 'bin_2',
            'bhz_2', 'bhr_2', 'bhn_2', 'onnxMatMul_207', 'fc2bias', 'onnxMatMul_208',
            'fc3bias', 'onnxMatMul_209', 'fc4bias', 'x', 'h1', 'h2', 'fc1MatMul', 'fc1Add',
            'gru1_a_', 'gru1_a', 'gru1_b_', 'gru1_b', 'gru1_c_', 'gru1_c', 'gru1_d_', 'gru1_d',
            'gru1_e_', 'gru1_e', 'gru1_f_', 'gru1_f', 'gru1_r_', 'gru1_r', 'gru1_z_', 'gru1_z',
            'gru1_n1', 'gru1_n2', 'gru1_n', 'gru1_hn1', 'gru1_hn2', 'gru1_hn3', 'rnn1GRU',
            'gru2_a_', 'gru2_a', 'gru2_b_', 'gru2_b', 'gru2_c_', 'gru2_c', 'gru2_d_', 'gru2_d',
            'gru2_e_', 'gru2_e', 'gru2_f_', 'gru2_f', 'gru2_r_', 'gru2_r', 'gru2_z_', 'gru2_z',
            'gru2_n1', 'gru2_n2', 'gru2_n', 'gru2_hn1', 'gru2_hn2', 'gru2_hn3', 'rnn2GRU',
            'fc2MatMul', 'fc2Add', 'relu', 'fc3MatMul', 'fc3Add', 'relu_1', 'fc4MatMul',
            'fc4Add', 'sigmoid'
        ]

        self.calib_dict = {key: [] for key in keys}
    
    def get_max(self, tensor):
        return np.max(tensor)
    
    def get_min(self, tensor):
        return np.min(tensor)




class NsNet2_npy(torch.nn.Module):
    def __init__(self, numpy_weights_path):
        super(NsNet2_npy, self).__init__()

        calib = Calibration()

        self.onnxMatMul_166 = np.load(numpy_weights_path + 'onnx__MatMul_166.npy').transpose()
        if not(calib.weights_calibrated):
            calib.calib_dict['onnxMatMul_166'].extend(self.onnxMatMul_166)
        
        self.fc1bias = np.load(numpy_weights_path + 'fc1_bias.npy')
        if not(calib.weights_calibrated):
            calib.calib_dict['fc1bias'].extend(self.fc1bias)

        self.onnxGRU_184 = np.load(numpy_weights_path + 'onnx__GRU_184.npy')
        
        self.Wiz_1 = self.onnxGRU_184[:,:400,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Wiz_1'].extend(self.Wiz_1)
        
        self.Wir_1 = self.onnxGRU_184[:,400:800,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Wir_1'].extend(self.Wir_1)

        self.Win_1 = self.onnxGRU_184[:,800:,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Win_1'].extend(self.Win_1)

        self.onnxGRU_185 = np.load(numpy_weights_path + 'onnx__GRU_185.npy')

        self.Whz_1 = self.onnxGRU_185[:,:400,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whz_1'].extend(self.Whz_1)

        self.Whr_1 = self.onnxGRU_185[:,400:800,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whr_1'].extend(self.Whr_1)

        self.Whn_1 = self.onnxGRU_185[:,800:,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whn_1'].extend(self.Whn_1)

        self.onnxGRU_186 = np.load(numpy_weights_path + 'onnx__GRU_186.npy')
        
        self.biz_1 = self.onnxGRU_186[:,:400]
        if not(calib.weights_calibrated):
            calib.calib_dict['biz_1'].extend(self.biz_1)

        self.bir_1 = self.onnxGRU_186[:,400:800]
        if not(calib.weights_calibrated):
            calib.calib_dict['bir_1'].extend(self.bir_1)

        self.bin_1 = self.onnxGRU_186[:,800:1200]
        if not(calib.weights_calibrated):
            calib.calib_dict['bin_1'].extend(self.bin_1)

        self.bhz_1 = self.onnxGRU_186[:,1200:1600]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhz_1'].extend(self.bhz_1)

        self.bhr_1 = self.onnxGRU_186[:,1600:2000]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhr_1'].extend(self.bhr_1)

        self.bhn_1 = self.onnxGRU_186[:,2000:]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhn_1'].extend(self.bhn_1)

        self.onnxGRU_204 = np.load(numpy_weights_path + 'onnx__GRU_204.npy')

        self.Wiz_2 = self.onnxGRU_204[:,:400,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Wiz_2'].extend(self.Wiz_2)

        self.Wir_2 = self.onnxGRU_204[:,400:800,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Wir_2'].extend(self.Wir_2)

        self.Win_2 = self.onnxGRU_204[:,800:,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Win_2'].extend(self.Win_2)

        self.onnxGRU_205 = np.load(numpy_weights_path + 'onnx__GRU_205.npy')

        self.Whz_2 = self.onnxGRU_205[:,:400,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whz_2'].extend(self.Whz_2)

        self.Whr_2 = self.onnxGRU_205[:,400:800,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whr_2'].extend(self.Whr_2)

        self.Whn_2 = self.onnxGRU_205[:,800:,:]
        if not(calib.weights_calibrated):
            calib.calib_dict['Whn_2'].extend(self.Whn_2)

        self.onnxGRU_206 = np.load(numpy_weights_path + 'onnx__GRU_206.npy')
        
        self.biz_2 = self.onnxGRU_206[:,:400]
        if not(calib.weights_calibrated):
            calib.calib_dict['biz_2'].extend(self.biz_2)
        
        self.bir_2 = self.onnxGRU_206[:,400:800]
        if not(calib.weights_calibrated):
            calib.calib_dict['bir_2'].extend(self.bir_2)

        self.bin_2 = self.onnxGRU_206[:,800:1200]
        if not(calib.weights_calibrated):
            calib.calib_dict['bin_2'].extend(self.bin_2)

        self.bhz_2 = self.onnxGRU_206[:,1200:1600]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhz_2'].extend(self.bhz_2)

        self.bhr_2 = self.onnxGRU_206[:,1600:2000]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhr_2'].extend(self.bhr_2)

        self.bhn_2 = self.onnxGRU_206[:,2000:]
        if not(calib.weights_calibrated):
            calib.calib_dict['bhn_2'].extend(self.bhn_2)

        self.onnxMatMul_207 = np.load(numpy_weights_path + 'onnx__MatMul_207.npy').transpose()
        if not(calib.weights_calibrated):
            calib.calib_dict['onnxMatMul_207'].extend(self.onnxMatMul_207)
        
        self.fc2bias = np.load(numpy_weights_path + 'fc2_bias.npy')
        if not(calib.weights_calibrated):
            calib.calib_dict['fc2bias'].extend(self.fc2bias)

        self.onnxMatMul_208 = np.load(numpy_weights_path + 'onnx__MatMul_208.npy').transpose()
        if not(calib.weights_calibrated):
            calib.calib_dict['onnxMatMul_208'].extend(self.onnxMatMul_208)
        
        self.fc3bias = np.load(numpy_weights_path + 'fc3_bias.npy')
        if not(calib.weights_calibrated):
            calib.calib_dict['fc3bias'].extend(self.fc3bias)

        self.onnxMatMul_209 = np.load(numpy_weights_path + 'onnx__MatMul_209.npy').transpose()
        if not(calib.weights_calibrated):
            calib.calib_dict['onnxMatMul_209'].extend(self.onnxMatMul_209)
        
        self.fc4bias = np.load(numpy_weights_path + 'fc4_bias.npy')
        if not(calib.weights_calibrated):
            calib.calib_dict['fc4bias'].extend(self.fc4bias)
        
        if not(calib.weights_calibrated):
            calib.weights_calibrated = True
        
        self.calib = calib
    
    def forward(self, x, h1, h2):
        x = x.squeeze()
        self.calib.calib_dict['x'].extend(x)

        h1 = h1.squeeze()
        self.calib.calib_dict['h1'].extend(h1)

        h2 = h2.squeeze()
        self.calib.calib_dict['h2'].extend(h2)

        # fully connected 1
        fc1MatMul = np.matmul(self.onnxMatMul_166, x)
        self.calib.calib_dict['fc1MatMul'].extend(fc1MatMul)

        fc1Add = np.add(fc1MatMul, self.fc1bias)
        self.calib.calib_dict['fc1Add'].extend(fc1Add)

        # gru 1
        gru1_a_ = np.matmul(self.Wir_1, fc1Add)
        self.calib.calib_dict['gru1_a_'].extend(gru1_a_)

        gru1_a = np.add(gru1_a_, self.bir_1)
        self.calib.calib_dict['gru1_a'].extend(gru1_a)

        gru1_b_ = np.matmul(self.Whr_1, h1)
        self.calib.calib_dict['gru1_b_'].extend(gru1_b_)

        gru1_b = np.add(gru1_b_, self.bhr_1)
        self.calib.calib_dict['gru1_b'].extend(gru1_b)

        gru1_c_ = np.matmul(self.Wiz_1, fc1Add)
        self.calib.calib_dict['gru1_c_'].extend(gru1_c_)

        gru1_c = np.add(gru1_c_, self.biz_1)
        self.calib.calib_dict['gru1_c'].extend(gru1_c)

        gru1_d_ = np.matmul(self.Whz_1, h1)
        self.calib.calib_dict['gru1_d_'].extend(gru1_d_)

        gru1_d = np.add(gru1_d_, self.bhz_1)
        self.calib.calib_dict['gru1_d'].extend(gru1_d)

        gru1_e_ = np.matmul(self.Win_1, fc1Add)
        self.calib.calib_dict['gru1_e_'].extend(gru1_e_)

        gru1_e = np.add(gru1_e_, self.bin_1)
        self.calib.calib_dict['gru1_e'].extend(gru1_e)

        gru1_f_ = np.matmul(self.Whn_1, h1)
        self.calib.calib_dict['gru1_f_'].extend(gru1_f_)

        gru1_f = np.add(gru1_f_, self.bhn_1)
        self.calib.calib_dict['gru1_f'].extend(gru1_f)

        gru1_r_ = np.add(gru1_a, gru1_b)
        self.calib.calib_dict['gru1_r_'].extend(gru1_r_)

        gru1_r = 1 / (1 + np.exp(-gru1_r_))
        self.calib.calib_dict['gru1_r'].extend(gru1_r)

        gru1_z_ = np.add(gru1_c, gru1_d)
        self.calib.calib_dict['gru1_z_'].extend(gru1_z_)

        gru1_z = 1 / (1 + np.exp(-gru1_z_))
        self.calib.calib_dict['gru1_z'].extend(gru1_z)

        gru1_n1 = gru1_r * gru1_f
        self.calib.calib_dict['gru1_n1'].extend(gru1_n1)

        gru1_n2 = np.add(gru1_e, gru1_n1)
        self.calib.calib_dict['gru1_n2'].extend(gru1_n2)

        gru1_n = np.tanh(gru1_n2)
        self.calib.calib_dict['gru1_n'].extend(gru1_n)

        gru1_hn1 = 1 - gru1_z
        self.calib.calib_dict['gru1_hn1'].extend(gru1_hn1)

        gru1_hn2 = gru1_hn1 * gru1_n
        self.calib.calib_dict['gru1_hn2'].extend(gru1_hn2)

        gru1_hn3 = gru1_z * h1
        self.calib.calib_dict['gru1_hn3'].extend(gru1_hn3)

        rnn1GRU = np.add(gru1_hn2, gru1_hn3)
        self.calib.calib_dict['rnn1GRU'].extend(rnn1GRU)

        # gru 2
        rnn1GRU = rnn1GRU.squeeze()

        gru2_a_ = np.matmul(self.Wir_2, rnn1GRU)
        self.calib.calib_dict['gru2_a_'].extend(gru2_a_)

        gru2_a = np.add(gru2_a_, self.bir_2)
        self.calib.calib_dict['gru2_a'].extend(gru2_a)

        gru2_b_ = np.matmul(self.Whr_2, h2)
        self.calib.calib_dict['gru2_b_'].extend(gru2_b_)

        gru2_b = np.add(gru2_b_, self.bhr_2)
        self.calib.calib_dict['gru2_b'].extend(gru2_b)

        gru2_c_ = np.matmul(self.Wiz_2, rnn1GRU)
        self.calib.calib_dict['gru2_c_'].extend(gru2_c_)

        gru2_c = np.add(gru2_c_, self.biz_2)
        self.calib.calib_dict['gru2_c'].extend(gru2_c)

        gru2_d_ = np.matmul(self.Whz_2, h2)
        self.calib.calib_dict['gru2_d_'].extend(gru2_d_)

        gru2_d = np.add(gru2_d_, self.bhz_2)
        self.calib.calib_dict['gru2_d'].extend(gru2_d)

        gru2_e_ = np.matmul(self.Win_2, rnn1GRU)
        self.calib.calib_dict['gru2_e_'].extend(gru2_e_)

        gru2_e = np.add(gru2_e_, self.bin_2)
        self.calib.calib_dict['gru2_e'].extend(gru2_e)

        gru2_f_ = np.matmul(self.Whn_2, h2)
        self.calib.calib_dict['gru2_f_'].extend(gru2_f_)

        gru2_f = np.add(gru2_f_, self.bhn_2)
        self.calib.calib_dict['gru2_f'].extend(gru2_f)

        gru2_r_ = np.add(gru2_a, gru2_b)
        self.calib.calib_dict['gru2_r_'].extend(gru2_r_)

        gru2_r = 1 / (1 + np.exp(-gru2_r_))
        self.calib.calib_dict['gru2_r'].extend(gru2_r)

        gru2_z_ = np.add(gru2_c, gru2_d)
        self.calib.calib_dict['gru2_z_'].extend(gru2_z_)

        gru2_z = 1 / (1 + np.exp(-gru2_z_))
        self.calib.calib_dict['gru2_z'].extend(gru2_z)

        gru2_n1 = gru2_r * gru2_f
        self.calib.calib_dict['gru2_n1'].extend(gru2_n1)

        gru2_n2 = np.add(gru2_e, gru2_n1)
        self.calib.calib_dict['gru2_n2'].extend(gru2_n2)

        gru2_n = np.tanh(gru2_n2)
        self.calib.calib_dict['gru2_n'].extend(gru2_n)

        gru2_hn1 = 1 - gru2_z
        self.calib.calib_dict['gru2_hn1'].extend(gru2_hn1)

        gru2_hn2 = gru2_hn1 * gru2_n
        self.calib.calib_dict['gru2_hn2'].extend(gru2_hn2)

        gru2_hn3 = gru2_z * h2
        self.calib.calib_dict['gru2_hn3'].extend(gru2_hn3)

        rnn2GRU = np.add(gru2_hn2, gru2_hn3)
        self.calib.calib_dict['rnn2GRU'].extend(rnn2GRU)

        rnn2GRU = rnn2GRU.squeeze()

        # fully connected 2
        fc2MatMul = np.matmul(self.onnxMatMul_207, rnn2GRU)
        self.calib.calib_dict['fc2MatMul'].extend(fc2MatMul)

        fc2Add = np.add(fc2MatMul, self.fc2bias)
        self.calib.calib_dict['fc2Add'].extend(fc2Add)

        relu = np.maximum(0, fc2Add)
        self.calib.calib_dict['relu'].extend(relu)

        # fully connected 3
        fc3MatMul = np.matmul(self.onnxMatMul_208, relu)
        self.calib.calib_dict['fc3MatMul'].extend(fc3MatMul)

        fc3Add = np.add(fc3MatMul, self.fc3bias)
        self.calib.calib_dict['fc3Add'].extend(fc3Add)

        relu_1 = np.maximum(0, fc3Add)
        self.calib.calib_dict['relu_1'].extend(relu_1)

        # fully connected 4
        fc4MatMul = np.matmul(self.onnxMatMul_209, relu_1)
        self.calib.calib_dict['fc4MatMul'].extend(fc4MatMul)

        fc4Add = np.add(fc4MatMul, self.fc4bias)
        self.calib.calib_dict['fc4Add'].extend(fc4Add)

        sigmoid = 1 / (1 + np.exp(-fc4Add))
        self.calib.calib_dict['sigmoid'].extend(sigmoid)

        return sigmoid, rnn1GRU, rnn2GRU