import numpy as np
import nsnet2.pytorch.nsnet2_q.calibration_output as co

class CalibrationParam():
    def __init__(self, bitwidth, signed, minimum, maximum, macro_prefix) -> None:
        self.bitwidth = bitwidth
        self.signed = signed
        self.minimum = minimum
        self.maximum = maximum
        self.macro_prefix = macro_prefix
    
    def S(self):
        if self.signed:
            return (self.maximum - self.minimum) / (2 ** (self.bitwidth-1) - 1)
        else:
            return (self.maximum - self.minimum) / (2 ** (self.bitwidth) - 1)
    
    def Z(self):
        if self.signed:
            return 0
        else:
            return np.round(-(self.minimum / self.S()))
    

def init_calibration(mpq_config):
    calib = {}
    calib['x'] = CalibrationParam(mpq_config[0], False, co.x_min, co.x_max, 'X_')
    calib['onnxMatMul_166'] = CalibrationParam(mpq_config[1], False, co.onnxMatMul_166_min, co.onnxMatMul_166_max, 'ONNX__MATMUL_166_')
    calib['fc1MatMul'] = CalibrationParam(mpq_config[2], False, co.fc1MatMul_min, co.fc1MatMul_max, 'FC1MATMUL_')
    calib['fc1bias'] = CalibrationParam(mpq_config[3], False, co.fc1bias_min, co.fc1bias_max, 'FC1_BIAS_')
    calib['fc1Add'] = CalibrationParam(mpq_config[4], False, co.fc1Add_min, co.fc1Add_max, 'FC1ADD_')
    calib['Wir_1'] = CalibrationParam(mpq_config[5], False, co.Wir_1_min, co.Wir_1_max, 'WIR_1_')
    calib['gru1_a_'] = CalibrationParam(mpq_config[6], False, co.gru1_a__min, co.gru1_a__max, 'GRU1_A__')
    calib['bir_1'] = CalibrationParam(mpq_config[7], False, co.bir_1_min, co.bir_1_max, 'BIR_1_')
    calib['gru1_a'] = CalibrationParam(mpq_config[8], False, co.gru1_a_min, co.gru1_a_max, 'GRU1_A_')
    calib['h1'] = CalibrationParam(mpq_config[9], False, co.h1_min, co.h1_max, 'H1_')
    calib['h2'] = CalibrationParam(mpq_config[10], False, co.h2_min, co.h2_max, 'H2_')
    calib['gru1_b_'] = CalibrationParam(mpq_config[11], False, co.gru1_b__min, co.gru1_b__max, 'GRU1_B__')
    calib['Wiz_1'] = CalibrationParam(mpq_config[12], False, co.Wiz_1_min, co.Wiz_1_max, 'WIZ_1_')
    calib['Win_1'] = CalibrationParam(mpq_config[13], False, co.Win_1_min, co.Win_1_max, 'WIN_1_')
    calib['Whz_1'] = CalibrationParam(mpq_config[14], False, co.Whz_1_min, co.Whz_1_max, 'WHZ_1_')
    calib['Whr_1'] = CalibrationParam(mpq_config[15], False, co.Whr_1_min, co.Whr_1_max, 'WHR_1_')
    calib['Whn_1'] = CalibrationParam(mpq_config[16], False, co.Whn_1_min, co.Whn_1_max, 'WHN_1_')
    calib['biz_1'] = CalibrationParam(mpq_config[17], False, co.biz_1_min, co.biz_1_max, 'BIZ_1_')
    calib['bin_1'] = CalibrationParam(mpq_config[18], False, co.bin_1_min, co.bin_1_max, 'BIN_1_')
    calib['bhz_1'] = CalibrationParam(mpq_config[19], False, co.bhz_1_min, co.bhz_1_max, 'BHZ_1_')
    calib['bhr_1'] = CalibrationParam(mpq_config[20], False, co.bhr_1_min, co.bhr_1_max, 'BHR_1_')
    calib['bhn_1'] = CalibrationParam(mpq_config[21], False, co.bhn_1_min, co.bhn_1_max, 'BHN_1_')
    calib['Wiz_2'] = CalibrationParam(mpq_config[22], False, co.Wiz_2_min, co.Wiz_2_max, 'WIZ_2_')
    calib['Wir_2'] = CalibrationParam(mpq_config[23], False, co.Wir_2_min, co.Wir_2_max, 'WIR_2_')
    calib['Win_2'] = CalibrationParam(mpq_config[24], False, co.Win_2_min, co.Win_2_max, 'WIN_2_')
    calib['Whz_2'] = CalibrationParam(mpq_config[25], False, co.Whz_2_min, co.Whz_2_max, 'WHZ_2_')
    calib['Whr_2'] = CalibrationParam(mpq_config[26], False, co.Whr_2_min, co.Whr_2_max, 'WHR_2_')
    calib['Whn_2'] = CalibrationParam(mpq_config[27], False, co.Whn_2_min, co.Whn_2_max, 'WHN_2_')
    calib['biz_2'] = CalibrationParam(mpq_config[28], False, co.biz_2_min, co.biz_2_max, 'BIZ_2_')
    calib['bir_2'] = CalibrationParam(mpq_config[29], False, co.bir_2_min, co.bir_2_max, 'BIR_2_')
    calib['bin_2'] = CalibrationParam(mpq_config[30], False, co.bin_2_min, co.bin_2_max, 'BIN_2_')
    calib['bhz_2'] = CalibrationParam(mpq_config[31], False, co.bhz_2_min, co.bhz_2_max, 'BHZ_2_')
    calib['bhr_2'] = CalibrationParam(mpq_config[32], False, co.bhr_2_min, co.bhr_2_max, 'BHR_2_')
    calib['bhn_2'] = CalibrationParam(mpq_config[33], False, co.bhn_2_min, co.bhn_2_max, 'BHN_2_')
    calib['onnxMatMul_207'] = CalibrationParam(mpq_config[34], False, co.onnxMatMul_207_min, co.onnxMatMul_207_max, 'ONNX__MATMUL_207_')
    calib['fc2bias'] = CalibrationParam(mpq_config[35], False, co.fc2bias_min, co.fc2bias_max, 'FC2_BIAS_')
    calib['onnxMatMul_208'] = CalibrationParam(mpq_config[36], False, co.onnxMatMul_208_min, co.onnxMatMul_208_max, 'ONNX__MATMUL_208_')
    calib['fc3bias'] = CalibrationParam(mpq_config[37], False, co.fc3bias_min, co.fc3bias_max, 'FC3_BIAS_')
    calib['onnxMatMul_209'] = CalibrationParam(mpq_config[38], False, co.onnxMatMul_209_min, co.onnxMatMul_209_max, 'ONNX__MATMUL_209_')
    calib['fc4bias'] = CalibrationParam(mpq_config[39], False, co.fc4bias_min, co.fc4bias_max, 'FC4_BIAS_')
    calib['gru1_b_'] = CalibrationParam(mpq_config[40], False, co.gru1_b__min, co.gru1_b__max, 'GRU1_B__')
    calib['gru1_b'] = CalibrationParam(mpq_config[41], False, co.gru1_b_min, co.gru1_b_max, 'GRU1_B_')
    calib['gru1_c_'] = CalibrationParam(mpq_config[42], False, co.gru1_c__min, co.gru1_c__max, 'GRU1_C__')
    calib['gru1_c'] = CalibrationParam(mpq_config[43], False, co.gru1_c_min, co.gru1_c_max, 'GRU1_C_')
    calib['gru1_d_'] = CalibrationParam(mpq_config[44], False, co.gru1_d__min, co.gru1_d__max, 'GRU1_D__')
    calib['gru1_d'] = CalibrationParam(mpq_config[45], False, co.gru1_d_min, co.gru1_d_max, 'GRU1_D_')
    calib['gru1_e_'] = CalibrationParam(mpq_config[46], False, co.gru1_e__min, co.gru1_e__max, 'GRU1_E__')
    calib['gru1_e'] = CalibrationParam(mpq_config[47], False, co.gru1_e_min, co.gru1_e_max, 'GRU1_E_')
    calib['gru1_f_'] = CalibrationParam(mpq_config[48], False, co.gru1_f__min, co.gru1_f__max, 'GRU1_F__')
    calib['gru1_f'] = CalibrationParam(mpq_config[49], False, co.gru1_f_min, co.gru1_f_max, 'GRU1_F_')
    calib['gru1_r_'] = CalibrationParam(mpq_config[50], False, co.gru1_r__min, co.gru1_r__max, 'GRU1_R__')
    calib['gru1_r'] = CalibrationParam(mpq_config[51], False, co.gru1_r_min, co.gru1_r_max, 'GRU1_R_')
    calib['gru1_z_'] = CalibrationParam(mpq_config[52], False, co.gru1_z__min, co.gru1_z__max, 'GRU1_Z__')
    calib['gru1_z'] = CalibrationParam(mpq_config[53], False, co.gru1_z_min, co.gru1_z_max, 'GRU1_Z_')
    calib['gru1_n1'] = CalibrationParam(mpq_config[54], False, co.gru1_n1_min, co.gru1_n1_max, 'GRU1_N1_')
    calib['gru1_n2'] = CalibrationParam(mpq_config[55], False, co.gru1_n2_min, co.gru1_n2_max, 'GRU1_N2_')
    calib['gru1_n'] = CalibrationParam(mpq_config[56], False, co.gru1_n_min, co.gru1_n_max, 'GRU1_N_')
    calib['gru1_hn1'] = CalibrationParam(mpq_config[57], False, co.gru1_hn1_min, co.gru1_hn1_max, 'GRU1_HN1_')
    calib['gru1_hn2'] = CalibrationParam(mpq_config[58], False, co.gru1_hn2_min, co.gru1_hn2_max, 'GRU1_HN2_')
    calib['gru1_hn3'] = CalibrationParam(mpq_config[59], False, co.gru1_hn3_min, co.gru1_hn3_max, 'GRU1_HN3_')
    calib['rnn1GRU'] = CalibrationParam(mpq_config[60], False, co.rnn1GRU_min, co.rnn1GRU_max, 'RNN1GRU_')
    calib['gru2_a_'] = CalibrationParam(mpq_config[61], False, co.gru2_a__min, co.gru2_a__max, 'GRU2_A__')
    calib['gru2_a'] = CalibrationParam(mpq_config[62], False, co.gru2_a_min, co.gru2_a_max, 'GRU2_A_')
    calib['gru2_b_'] = CalibrationParam(mpq_config[63], False, co.gru2_b__min, co.gru2_b__max, 'GRU2_B__')
    calib['gru2_b'] = CalibrationParam(mpq_config[64], False, co.gru2_b_min, co.gru2_b_max, 'GRU2_B_')
    calib['gru2_c_'] = CalibrationParam(mpq_config[65], False, co.gru2_c__min, co.gru2_c__max, 'GRU2_C__')
    calib['gru2_c'] = CalibrationParam(mpq_config[66], False, co.gru2_c_min, co.gru2_c_max, 'GRU2_C_')
    calib['gru2_d_'] = CalibrationParam(mpq_config[67], False, co.gru2_d__min, co.gru2_d__max, 'GRU2_D__')
    calib['gru2_d'] = CalibrationParam(mpq_config[68], False, co.gru2_d_min, co.gru2_d_max, 'GRU2_D_')
    calib['gru2_e_'] = CalibrationParam(mpq_config[69], False, co.gru2_e__min, co.gru2_e__max, 'GRU2_E__')
    calib['gru2_e'] = CalibrationParam(mpq_config[70], False, co.gru2_e_min, co.gru2_e_max, 'GRU2_E_')
    calib['gru2_f_'] = CalibrationParam(mpq_config[71], False, co.gru2_f__min, co.gru2_f__max, 'GRU2_F__')
    calib['gru2_f'] = CalibrationParam(mpq_config[72], False, co.gru2_f_min, co.gru2_f_max, 'GRU2_F_')
    calib['gru2_r_'] = CalibrationParam(mpq_config[73], False, co.gru2_r__min, co.gru2_r__max, 'GRU2_R__')
    calib['gru2_r'] = CalibrationParam(mpq_config[74], False, co.gru2_r_min, co.gru2_r_max, 'GRU2_R_')
    calib['gru2_z_'] = CalibrationParam(mpq_config[75], False, co.gru2_z__min, co.gru2_z__max, 'GRU2_Z__')
    calib['gru2_z'] = CalibrationParam(mpq_config[76], False, co.gru2_z_min, co.gru2_z_max, 'GRU2_Z_')
    calib['gru2_n1'] = CalibrationParam(mpq_config[77], False, co.gru2_n1_min, co.gru2_n1_max, 'GRU2_N1_')
    calib['gru2_n2'] = CalibrationParam(mpq_config[78], False, co.gru2_n2_min, co.gru2_n2_max, 'GRU2_N2_')
    calib['gru2_n'] = CalibrationParam(mpq_config[79], False, co.gru2_n_min, co.gru2_n_max, 'GRU2_N_')
    calib['gru2_hn1'] = CalibrationParam(mpq_config[80], False, co.gru2_hn1_min, co.gru2_hn1_max, 'GRU2_HN1_')
    calib['gru2_hn2'] = CalibrationParam(mpq_config[81], False, co.gru2_hn2_min, co.gru2_hn2_max, 'GRU2_HN2_')
    calib['gru2_hn3'] = CalibrationParam(mpq_config[82], False, co.gru2_hn3_min, co.gru2_hn3_max, 'GRU2_HN3_')
    calib['rnn2GRU'] = CalibrationParam(mpq_config[83], False, co.rnn2GRU_min, co.rnn2GRU_max, 'RNN2GRU_')
    calib['fc2MatMul'] = CalibrationParam(mpq_config[84], False, co.fc2MatMul_min, co.fc2MatMul_max, 'FC2MATMUL_')
    calib['fc2Add'] = CalibrationParam(mpq_config[85], False, co.fc2Add_min, co.fc2Add_max, 'FC2ADD_')
    calib['relu'] = CalibrationParam(mpq_config[86], False, co.relu_min, co.relu_max, 'RELU_')
    calib['fc3MatMul'] = CalibrationParam(mpq_config[87], False, co.fc3MatMul_min, co.fc3MatMul_max, 'FC3MATMUL_')
    calib['fc3Add'] = CalibrationParam(mpq_config[88], False, co.fc3Add_min, co.fc3Add_max, 'FC3ADD_')
    calib['relu_1'] = CalibrationParam(mpq_config[89], False, co.relu_1_min, co.relu_1_max, 'RELU_1_')
    calib['fc4MatMul'] = CalibrationParam(mpq_config[90], False, co.fc4MatMul_min, co.fc4MatMul_max, 'FC4MATMUL_')
    calib['fc4Add'] = CalibrationParam(mpq_config[91], False, co.fc4Add_min, co.fc4Add_max, 'FC4ADD_')
    calib['sigmoid'] = CalibrationParam(mpq_config[92], False, co.sigmoid_min, co.sigmoid_max, 'SIGMOID_')
    return calib