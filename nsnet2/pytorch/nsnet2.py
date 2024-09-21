import torch
import numpy as np

class NsNet2_npy(torch.nn.Module):
    def __init__(self):
        super(NsNet2_npy, self).__init__()
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
        torch.set_printoptions(precision=8)
        x = x.squeeze()
        h1 = h1.squeeze()
        h2 = h2.squeeze()

        # fully connected 1
        fc1MatMul = np.matmul(self.onnxMatMul_166, x)
        fc1Add = np.add(fc1MatMul, self.fc1bias)
        print(f"max: {max(self.fc1bias)}")
        print(f"min: {min(self.fc1bias)}")
        
        # gru 1
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

class NsNet2(torch.nn.Module):
    def __init__(self):
        super(NsNet2, self).__init__()

        self.onnxMatMul_166 = torch.load('tensor_onnxMatMul_166.pt').transpose(0, 1)
        self.fc1bias = torch.load('tensor_fc1bias.pt')

        self.onnxGRU_184 = torch.load('tensor_onnxGRU_184.pt')
        self.Wiz_1 = self.onnxGRU_184[:,:400,:]
        self.Wir_1 = self.onnxGRU_184[:,400:800,:]
        self.Win_1 = self.onnxGRU_184[:,800:,:]

        self.onnxGRU_185 = torch.load('tensor_onnxGRU_185.pt')
        self.Whz_1 = self.onnxGRU_185[:,:400,:]
        self.Whr_1 = self.onnxGRU_185[:,400:800,:]
        self.Whn_1 = self.onnxGRU_185[:,800:,:]

        self.onnxGRU_186 = torch.load('tensor_onnxGRU_186.pt')
        self.biz_1 = self.onnxGRU_186[:,:400]
        self.bir_1 = self.onnxGRU_186[:,400:800]
        self.bin_1 = self.onnxGRU_186[:,800:1200]
        self.bhz_1 = self.onnxGRU_186[:,1200:1600]
        self.bhr_1 = self.onnxGRU_186[:,1600:2000]
        self.bhn_1 = self.onnxGRU_186[:,2000:]

        self.onnxGRU_204 = torch.load('tensor_onnxGRU_204.pt')
        self.Wiz_2 = self.onnxGRU_204[:,:400,:]
        self.Wir_2 = self.onnxGRU_204[:,400:800,:]
        self.Win_2 = self.onnxGRU_204[:,800:,:]

        self.onnxGRU_205 = torch.load('tensor_onnxGRU_205.pt')
        self.Whz_2 = self.onnxGRU_205[:,:400,:]
        self.Whr_2 = self.onnxGRU_205[:,400:800,:]
        self.Whn_2 = self.onnxGRU_205[:,800:,:]

        self.onnxGRU_206 = torch.load('tensor_onnxGRU_206.pt')
        self.biz_2 = self.onnxGRU_206[:,:400]
        self.bir_2 = self.onnxGRU_206[:,400:800]
        self.bin_2 = self.onnxGRU_206[:,800:1200]
        self.bhz_2 = self.onnxGRU_206[:,1200:1600]
        self.bhr_2 = self.onnxGRU_206[:,1600:2000]
        self.bhn_2 = self.onnxGRU_206[:,2000:]

        self.onnxMatMul_207 = torch.load('tensor_onnxMatMul_207.pt').transpose(0, 1)
        self.fc2bias = torch.load('tensor_fc2bias.pt')

        self.onnxMatMul_208 = torch.load('tensor_onnxMatMul_208.pt').transpose(0, 1)
        self.fc3bias = torch.load('tensor_fc3bias.pt')

        self.onnxMatMul_209 = torch.load('tensor_onnxMatMul_209.pt').transpose(0, 1)
        self.fc4bias = torch.load('tensor_fc4bias.pt')

    def forward(self, x, h1, h2):
        torch.set_printoptions(precision=8)
        x = x.squeeze()
        h1 = h1.squeeze()
        h2 = h2.squeeze()

        # fully connected 1
        fc1MatMul = torch.matmul(self.onnxMatMul_166, x)
        fc1Add = torch.add(fc1MatMul, self.fc1bias)
        
        # gru 1
        gru1_a_ = torch.matmul(self.Wir_1, fc1Add)
        gru1_a = torch.add(gru1_a_, self.bir_1)
        gru1_b_ = torch.matmul(self.Whr_1, h1)
        gru1_b = torch.add(gru1_b_, self.bhr_1)
        gru1_c_ = torch.matmul(self.Wiz_1, fc1Add)
        gru1_c = torch.add(gru1_c_, self.biz_1)
        gru1_d_ = torch.matmul(self.Whz_1, h1)
        gru1_d = torch.add(gru1_d_, self.bhz_1)
        gru1_e_ = torch.matmul(self.Win_1, fc1Add)
        gru1_e = torch.add(gru1_e_, self.bin_1)
        gru1_f_ = torch.matmul(self.Whn_1, h1)
        gru1_f = torch.add(gru1_f_, self.bhn_1)

        gru1_r_ = torch.add(gru1_a, gru1_b)
        gru1_r = torch.sigmoid(gru1_r_)

        gru1_z_ = torch.add(gru1_c, gru1_d)
        gru1_z = torch.sigmoid(gru1_z_)

        gru1_n1 = torch.mul(gru1_r, gru1_f)
        gru1_n2 = torch.add(gru1_e, gru1_n1)
        gru1_n = torch.tanh(gru1_n2)

        gru1_hn1 = torch.sub(1, gru1_z)
        gru1_hn2 = torch.mul(gru1_hn1, gru1_n)
        gru1_hn3 = torch.mul(gru1_z, h1)
        rnn1GRU = torch.add(gru1_hn2, gru1_hn3)

        # gru 2
        rnn1GRU = rnn1GRU.squeeze()
        gru2_a_ = torch.matmul(self.Wir_2, rnn1GRU)
        gru2_a = torch.add(gru2_a_, self.bir_2)
        gru2_b_ = torch.matmul(self.Whr_2, h2)
        gru2_b = torch.add(gru2_b_, self.bhr_2)
        gru2_c_ = torch.matmul(self.Wiz_2, rnn1GRU)
        gru2_c = torch.add(gru2_c_, self.biz_2)
        gru2_d_ = torch.matmul(self.Whz_2, h2)
        gru2_d = torch.add(gru2_d_, self.bhz_2)
        gru2_e_ = torch.matmul(self.Win_2, rnn1GRU)
        gru2_e = torch.add(gru2_e_, self.bin_2)
        gru2_f_ = torch.matmul(self.Whn_2, h2)
        gru2_f = torch.add(gru2_f_, self.bhn_2)

        gru2_r_ = torch.add(gru2_a, gru2_b)
        gru2_r = torch.sigmoid(gru2_r_)

        gru2_z_ = torch.add(gru2_c, gru2_d)
        gru2_z = torch.sigmoid(gru2_z_)

        gru2_n1 = torch.mul(gru2_r, gru2_f)
        gru2_n2 = torch.add(gru2_e, gru2_n1)
        gru2_n = torch.tanh(gru2_n2)

        gru2_hn1 = torch.sub(1, gru2_z)
        gru2_hn2 = torch.mul(gru2_hn1, gru2_n)
        gru2_hn3 = torch.mul(gru2_z, h2)
        rnn2GRU = torch.add(gru2_hn2, gru2_hn3)
        rnn2GRU = rnn2GRU.squeeze()

        # fully connected 2
        fc2MatMul = torch.matmul(self.onnxMatMul_207, rnn2GRU)
        fc2Add = torch.add(fc2MatMul, self.fc2bias)
        relu = torch.relu(fc2Add)

        # fully connected 3
        fc3MatMul = torch.matmul(self.onnxMatMul_208, relu)
        fc3Add = torch.add(fc3MatMul, self.fc3bias)
        relu_1 = torch.relu(fc3Add)

        # fully connected 4
        fc4MatMul = torch.matmul(self.onnxMatMul_209, relu_1)
        fc4Add = torch.add(fc4MatMul, self.fc4bias)
        sigmoid = torch.sigmoid(fc4Add)

        return sigmoid


#nsnet = NsNet2()
#x = torch.ones([1, 1, 257], dtype=torch.float32)
#h1 = torch.ones([1, 1, 400], dtype=torch.float32)*2
#h2 = torch.ones([1, 1, 400], dtype=torch.float32)
#output = nsnet(x, h1, h2)
#print(output)
