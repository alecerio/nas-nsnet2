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
        self.calib['bir_1'] = CalibrationParam(8, False, -0.07920225709676743, 0.20611026883125305)
        self.calib['gru1_a'] = CalibrationParam(8, False, -0.6046768426895142, 0.8871182203292847)
        self.calib['h1'] = CalibrationParam(8, False, -0.002647488145157695, 0.0028339680284261703)
        self.calib['h2'] = CalibrationParam(8, False, -0.0031016061548143625, 0.0030250486452132463)
        self.calib['gru1_b_'] = CalibrationParam(8, False, -0.004922409541904926, 0.004103424027562141)
        self.calib['Wiz_1'] = CalibrationParam(8, False, -0.43284985423088074, 0.46175122261047363)
        self.calib['Win_1'] = CalibrationParam(8, False, -0.3236880302429199, 0.39607325196266174)
        self.calib['Whz_1'] = CalibrationParam(8, False, -1.8417714834213257, 1.7173254489898682)
        self.calib['Whr_1'] = CalibrationParam(8, False, -1.1574513912200928, 1.0300449132919312)
        self.calib['Whn_1'] = CalibrationParam(8, False, -0.7756922245025635, 0.9530389308929443)
        self.calib['biz_1'] = CalibrationParam(8, False, -0.5063393712043762, 0.36664387583732605)
        self.calib['bin_1'] = CalibrationParam(8, False, -0.5539973378181458, 0.17938342690467834)
        self.calib['bhz_1'] = CalibrationParam(8, False, -0.5337516665458679, 0.4148772358894348)
        self.calib['bhr_1'] = CalibrationParam(8, False, -0.07688436657190323, 0.14814253151416779)
        self.calib['bhn_1'] = CalibrationParam(8, False, -0.7828555107116699, 0.9008108973503113)
        self.calib['Wiz_2'] = CalibrationParam(8, False, -0.9102030992507935, 0.9408696889877319)
        self.calib['Wir_2'] = CalibrationParam(8, False, -0.9560997486114502, 0.6683358550071716)
        self.calib['Win_2'] = CalibrationParam(8, False, -0.4721935987472534, 0.48561596870422363)
        self.calib['Whz_2'] = CalibrationParam(8, False, -1.2992678880691528, 1.2991048097610474)
        self.calib['Whr_2'] = CalibrationParam(8, False, -0.8318714499473572, 1.1085889339447021)
        self.calib['Whn_2'] = CalibrationParam(8, False, -0.955470085144043, 1.046797513961792)
        self.calib['biz_2'] = CalibrationParam(8, False, -0.44805487990379333, 0.1560053527355194)
        self.calib['bir_2'] = CalibrationParam(8, False, -0.08767592161893845, 0.11347303539514542)
        self.calib['bin_2'] = CalibrationParam(8, False, -0.239909827709198, 0.12033259868621826)
        self.calib['bhz_2'] = CalibrationParam(8, False, -0.43745461106300354, 0.12699371576309204)
        self.calib['bhr_2'] = CalibrationParam(8, False, -0.09617264568805695, 0.07690174877643585)
        self.calib['bhn_2'] = CalibrationParam(8, False, -0.17204178869724274, 0.19739042222499847)
        self.calib['onnxMatMul_207'] = CalibrationParam(8, False, -1.3657219409942627, 1.158295750617981)
        self.calib['fc2bias'] = CalibrationParam(8, False, -0.1750922054052353, 0.1385071724653244)
        self.calib['onnxMatMul_208'] = CalibrationParam(8, False, -3.1666038036346436, 2.5026357173919678)
        self.calib['fc3bias'] = CalibrationParam(8, False, -0.10188056528568268, 0.0899151861667633)
        self.calib['onnxMatMul_209'] = CalibrationParam(8, False, -1.300571084022522, 1.928941249847412)
        self.calib['fc4bias'] = CalibrationParam(8, False, -0.10699586570262909, 0.04597663879394531)
        self.calib['gru1_b_'] = CalibrationParam(8, False, -0.004922409541904926, 0.004103424027562141)
        self.calib['gru1_b'] = CalibrationParam(8, False, -0.07475128024816513, 0.14630259573459625)
        self.calib['gru1_c_'] = CalibrationParam(8, False, -1.5660111904144287, 1.0454494953155518)
        self.calib['gru1_c'] = CalibrationParam(8, False, -1.9779117107391357, 1.3700535297393799)
        self.calib['gru1_d_'] = CalibrationParam(8, False, -0.008429044857621193, 0.006403823848813772)
        self.calib['gru1_d'] = CalibrationParam(8, False, -0.5529640316963196, 0.41507571935653687)
        self.calib['gru1_e_'] = CalibrationParam(8, False, -1.3637111186981201, 1.018247127532959)
        self.calib['gru1_e'] = CalibrationParam(8, False, -1.8583884239196777, 1.1587692499160767)
        self.calib['gru1_f_'] = CalibrationParam(8, False, -0.005411399528384209, 0.0061536869034171104)
        self.calib['gru1_f'] = CalibrationParam(8, False, -0.7833794355392456, 0.8978855013847351)
        self.calib['gru1_r_'] = CalibrationParam(8, False, -0.6226556301116943, 0.9800534844398499)
        self.calib['gru1_r'] = CalibrationParam(8, False, 0.3491777181625366, 0.7271188497543335)
        self.calib['gru1_z_'] = CalibrationParam(8, False, -2.392332077026367, 1.7851293087005615)
        self.calib['gru1_z'] = CalibrationParam(8, False, 0.08375928550958633, 0.856329083442688)
        self.calib['gru1_n1'] = CalibrationParam(8, False, -0.5516456365585327, 0.4556601643562317)
        self.calib['gru1_n2'] = CalibrationParam(8, False, -2.4100341796875, 1.1423156261444092)
        self.calib['gru1_n'] = CalibrationParam(8, False, -0.983996570110321, 0.8151924014091492)
        self.calib['gru1_hn1'] = CalibrationParam(8, False, 0.143670916557312, 0.9162406921386719)
        self.calib['gru1_hn2'] = CalibrationParam(8, False, -0.9015777111053467, 0.45391541719436646)
        self.calib['gru1_hn3'] = CalibrationParam(8, False, -0.0016954239690676332, 0.0016671211924403906)
        self.calib['rnn1GRU'] = CalibrationParam(8, False, -0.9016271829605103, 0.4546271562576294)
        self.calib['gru2_a_'] = CalibrationParam(8, False, -0.681461751461029, 0.9113116264343262)
        self.calib['gru2_a'] = CalibrationParam(8, False, -0.7149235010147095, 0.908741295337677)
        self.calib['gru2_b_'] = CalibrationParam(8, False, -0.005148397758603096, 0.011014967225492)
        self.calib['gru2_b'] = CalibrationParam(8, False, -0.09436552226543427, 0.07623167335987091)
        self.calib['gru2_c_'] = CalibrationParam(8, False, -1.2279887199401855, 1.1102137565612793)
        self.calib['gru2_c'] = CalibrationParam(8, False, -1.6760436296463013, 1.200895071029663)
        self.calib['gru2_d_'] = CalibrationParam(8, False, -0.008141756057739258, 0.00894666276872158)
        self.calib['gru2_d'] = CalibrationParam(8, False, -0.43875735998153687, 0.1262134611606598)
        self.calib['gru2_e_'] = CalibrationParam(8, False, -0.6732944250106812, 0.6314664483070374)

        # weights

        # onnxMatMul_166
        self.onnxMatMul_166 = np.load('onnx__MatMul_166.npy').transpose()
        self.onnxMatMul_166_q = self._quantize_tensor(self.onnxMatMul_166, 'onnxMatMul_166')
        
        # fc1bias
        self.fc1bias = np.load('fc1_bias.npy')
        self.fc1bias_q = self._quantize_tensor(self.fc1bias, 'fc1bias')

        # Wiz_1, Wir_1, Win_1
        self.onnxGRU_184 = np.load('onnx__GRU_184.npy')
        self.Wiz_1 = self.onnxGRU_184[:,:400,:]
        self.Wir_1 = self.onnxGRU_184[:,400:800,:]
        self.Win_1 = self.onnxGRU_184[:,800:,:]

        self.Wiz_1_q = self._quantize_tensor(self.Wiz_1, 'Wiz_1')
        self.Wir_1_q = self._quantize_tensor(self.Wir_1, 'Wir_1')
        self.Win_1_q = self._quantize_tensor(self.Win_1, 'Win_1')

        self.onnxGRU_185 = np.load('onnx__GRU_185.npy')
        self.Whz_1 = self.onnxGRU_185[:,:400,:]
        self.Whr_1 = self.onnxGRU_185[:,400:800,:]
        self.Whn_1 = self.onnxGRU_185[:,800:,:]

        self.Whz_1_q = self._quantize_tensor(self.Whz_1, 'Whz_1')
        self.Whr_1_q = self._quantize_tensor(self.Whr_1, 'Whr_1')
        self.Whn_1_q = self._quantize_tensor(self.Whn_1, 'Whn_1')

        # biz_1, bir_1, bin_1, bhz_1, bhr_1, bhn_1
        self.onnxGRU_186 = np.load('onnx__GRU_186.npy')
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

        self.onnxGRU_204 = np.load('onnx__GRU_204.npy')
        self.Wiz_2 = self.onnxGRU_204[:,:400,:]
        self.Wir_2 = self.onnxGRU_204[:,400:800,:]
        self.Win_2 = self.onnxGRU_204[:,800:,:]

        self.Wiz_2_q = self._quantize_tensor(self.Wiz_2, 'Wiz_2')
        self.Wir_2_q = self._quantize_tensor(self.Wir_2, 'Wir_2')
        self.Win_2_q = self._quantize_tensor(self.Win_2, 'Win_2')

        self.onnxGRU_205 = np.load('onnx__GRU_205.npy')
        self.Whz_2 = self.onnxGRU_205[:,:400,:]
        self.Whr_2 = self.onnxGRU_205[:,400:800,:]
        self.Whn_2 = self.onnxGRU_205[:,800:,:]

        self.Whz_2_q = self._quantize_tensor(self.Whz_2, 'Whz_2')
        self.Whr_2_q = self._quantize_tensor(self.Whr_2, 'Whr_2')
        self.Whn_2_q = self._quantize_tensor(self.Whn_2, 'Whn_2')

        self.onnxGRU_206 = np.load('onnx__GRU_206.npy')
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

        self.onnxMatMul_207 = np.load('onnx__MatMul_207.npy').transpose()
        self.onnxMatMul_207_q = self._quantize_tensor(self.onnxMatMul_207, 'onnxMatMul_207')
        
        self.fc2bias = np.load('fc2_bias.npy')
        self.fc2bias_q = self._quantize_tensor(self.fc2bias, 'fc2bias')

        self.onnxMatMul_208 = np.load('onnx__MatMul_208.npy').transpose()
        self.onnxMatMul_208_q = self._quantize_tensor(self.onnxMatMul_208, 'onnxMatMul_208')
        
        self.fc3bias = np.load('fc3_bias.npy')
        self.fc3bias_q = self._quantize_tensor(self.fc3bias, 'fc3bias')

        self.onnxMatMul_209 = np.load('onnx__MatMul_209.npy').transpose()
        self.onnxMatMul_209_q = self._quantize_tensor(self.onnxMatMul_209, 'onnxMatMul_209')
        
        self.fc4bias = np.load('fc4_bias.npy')
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
        fc1MatMul = np.matmul(self.onnxMatMul_166, x) # to remove

        # fc1Add_q
        fc1Add_q = self._quantize_add(fc1MatMul_q, self.fc1bias_q, 'fc1MatMul', 'fc1bias', 'fc1Add')
        fc1Add = np.add(fc1MatMul, self.fc1bias) # to remove
        
        # gru1_a_
        gru1_a__q = self._quantize_matmul(self.Wir_1_q, fc1Add_q, 'Wir_1', 'fc1Add', 'gru1_a_')
        gru1_a_ = np.matmul(self.Wir_1, fc1Add) # to remove
        
        # gru1_a
        gru1_a_q = self._quantize_add(gru1_a__q, self.bir_1_q, 'gru1_a_', 'bir_1', 'gru1_a')
        gru1_a = np.add(gru1_a_, self.bir_1)

        # gru1_b_
        gru1_b__q = self._quantize_matmul(self.Whr_1_q, h1_q, 'Whr_1', 'h1', 'gru1_b_')
        gru1_b_ = np.matmul(self.Whr_1, h1)

        # gru1_b
        gru1_b_q = self._quantize_add(gru1_b__q, self.bhr_1_q, 'gru1_b_', 'bhr_1', 'gru1_b')
        gru1_b = np.add(gru1_b_, self.bhr_1)
        self._compare(gru1_b, gru1_b_q, self.calib['gru1_b'])

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
        gru1_r_q = self._quantize(temp_y, self.calib['gru1_r'].S(), self.calib['gru1_r'].Z())
        gru1_r = 1 / (1 + np.exp(-gru1_r_))

        # gru1_z_
        gru1_z__q = self._quantize_add(gru1_c_q, gru1_d_q, 'gru1_c', 'gru1_d', 'gru1_z_')
        gru1_z_ = np.add(gru1_c, gru1_d)
        
        # gru1_z
        temp_x = self._dequantize(gru1_z__q, self.calib['gru1_z_'].S(), self.calib['gru1_z_'].Z())
        temp_y = 1 / (1 + np.exp(-temp_x))
        gru1_z_q = self._quantize(temp_y, self.calib['gru1_z'].S(), self.calib['gru1_z'].Z())
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
        gru1_n_q = self._quantize(temp_y, self.calib['gru1_n'].S(), self.calib['gru1_n'].Z())
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
        self._compare(gru2_e_, gru2_e__q, self.calib['gru2_e_'])
        
        # gru2_e
        gru2_e = np.add(gru2_e_, self.bin_2)
        print(f"min: {np.min(gru2_e)}")
        print(f"max: {np.max(gru2_e)}")

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

    def _compare(self, tensor_f32, tensor_int, calib):
        print(np.mean(np.abs(tensor_f32 - self._dequantize(tensor_int, calib.S(), calib.Z()))))
    
    def _quantize_tensor(self, tensor_f32, c_key):
        c = self.calib[c_key]
        return self._quantize(tensor_f32, c.S(), c.Z())
    
    def _quantize_matmul(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        return np.round(
            (ca.S()*cb.S() / cy.S()) * np.matmul(A - ca.Z(), B - cb.Z()) + cy.Z()
        )
    
    def _quantize_add(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        return (ca.S() / cy.S()) * (A - ca.Z()) + (cb.S() / cy.S()) * (B - cb.Z()) + cy.Z()
    
    def _quantize_mul(self, A, B, ca_key, cb_key, cy_key):
        ca = self.calib[ca_key]
        cb = self.calib[cb_key]
        cy = self.calib[cy_key]
        return (ca.S() * cb.S() / cy.S()) * (A - ca.Z()) * (B - cb.Z()) + cy.Z()
    
    def _quantize_one_minus_x(self, X, cx_key, cy_key):
        cx = self.calib[cx_key]
        cy = self.calib[cy_key]
        one_q = self._quantize(np.ones(X.shape), cx.S(), cx.Z())
        return (cx.S() / cy.S()) * (one_q - X) + cy.Z()