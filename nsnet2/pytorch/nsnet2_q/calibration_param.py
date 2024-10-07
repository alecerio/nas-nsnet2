import numpy as np

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
    calib['x'] = CalibrationParam(mpq_config[0], False, -0.0025095, 0.0022181, 'X_')
    calib['onnxMatMul_166'] = CalibrationParam(mpq_config[1], False, -0.22075387835502625, 0.208940327167511, 'ONNX__MATMUL_166_')
    calib['fc1MatMul'] = CalibrationParam(mpq_config[2], False, -0.00291599917, 0.0017367251, 'FC1MATMUL_')
    calib['fc1bias'] = CalibrationParam(mpq_config[3], False, -0.48688140511512756, 0.5176185369491577, 'FC1_BIAS_')
    calib['fc1Add'] = CalibrationParam(mpq_config[4], False, -0.48778465390205383, 0.5181604027748108, 'FC1ADD_')
    calib['Wir_1'] = CalibrationParam(mpq_config[5], False, -0.34401071071624756, 0.29191476106643677, 'WIR_1_')
    calib['gru1_a_'] = CalibrationParam(mpq_config[6], False, -0.6389939785003662, 0.7715625762939453, 'GRU1_A__')
    calib['bir_1'] = CalibrationParam(mpq_config[7], False, -0.07920225709676743, 0.20611026883125305, 'BIR_1_')
    calib['gru1_a'] = CalibrationParam(mpq_config[8], False, -0.6046768426895142, 0.8871182203292847, 'GRU1_A_')
    calib['h1'] = CalibrationParam(mpq_config[9], False, -0.002647488145157695, 0.0028339680284261703, 'H1_')
    calib['h2'] = CalibrationParam(mpq_config[10], False, -0.0031016061548143625, 0.0030250486452132463, 'H2_')
    calib['gru1_b_'] = CalibrationParam(mpq_config[11], False, -0.004922409541904926, 0.004103424027562141, 'GRU1_B__')
    calib['Wiz_1'] = CalibrationParam(mpq_config[12], False, -0.43284985423088074, 0.46175122261047363, 'WIZ_1_')
    calib['Win_1'] = CalibrationParam(mpq_config[13], False, -0.3236880302429199, 0.39607325196266174, 'WIN_1_')
    calib['Whz_1'] = CalibrationParam(mpq_config[14], False, -1.8417714834213257, 1.7173254489898682, 'WHZ_1_')
    calib['Whr_1'] = CalibrationParam(mpq_config[15], False, -1.1574513912200928, 1.0300449132919312, 'WHR_1_')
    calib['Whn_1'] = CalibrationParam(mpq_config[16], False, -0.7756922245025635, 0.9530389308929443, 'WHN_1_')
    calib['biz_1'] = CalibrationParam(mpq_config[17], False, -0.5063393712043762, 0.36664387583732605, 'BIZ_1_')
    calib['bin_1'] = CalibrationParam(mpq_config[18], False, -0.5539973378181458, 0.17938342690467834, 'BIN_1_')
    calib['bhz_1'] = CalibrationParam(mpq_config[19], False, -0.5337516665458679, 0.4148772358894348, 'BHZ_1_')
    calib['bhr_1'] = CalibrationParam(mpq_config[20], False, -0.07688436657190323, 0.14814253151416779, 'BHR_1_')
    calib['bhn_1'] = CalibrationParam(mpq_config[21], False, -0.7828555107116699, 0.9008108973503113, 'BHN_1_')
    calib['Wiz_2'] = CalibrationParam(mpq_config[22], False, -0.9102030992507935, 0.9408696889877319, 'WIZ_2_')
    calib['Wir_2'] = CalibrationParam(mpq_config[23], False, -0.9560997486114502, 0.6683358550071716, 'WIR_2_')
    calib['Win_2'] = CalibrationParam(mpq_config[24], False, -0.4721935987472534, 0.48561596870422363, 'WIN_2_')
    calib['Whz_2'] = CalibrationParam(mpq_config[25], False, -1.2992678880691528, 1.2991048097610474, 'WHZ_2_')
    calib['Whr_2'] = CalibrationParam(mpq_config[26], False, -0.8318714499473572, 1.1085889339447021, 'WHR_2_')
    calib['Whn_2'] = CalibrationParam(mpq_config[27], False, -0.955470085144043, 1.046797513961792, 'WHN_2_')
    calib['biz_2'] = CalibrationParam(mpq_config[28], False, -0.44805487990379333, 0.1560053527355194, 'BIZ_2_')
    calib['bir_2'] = CalibrationParam(mpq_config[29], False, -0.08767592161893845, 0.11347303539514542, 'BIR_2_')
    calib['bin_2'] = CalibrationParam(mpq_config[30], False, -0.239909827709198, 0.12033259868621826, 'BIN_2_')
    calib['bhz_2'] = CalibrationParam(mpq_config[31], False, -0.43745461106300354, 0.12699371576309204, 'BHZ_2_')
    calib['bhr_2'] = CalibrationParam(mpq_config[32], False, -0.09617264568805695, 0.07690174877643585, 'BHR_2_')
    calib['bhn_2'] = CalibrationParam(mpq_config[33], False, -0.17204178869724274, 0.19739042222499847, 'BHN_2_')
    calib['onnxMatMul_207'] = CalibrationParam(mpq_config[34], False, -1.3657219409942627, 1.158295750617981, 'ONNX__MATMUL_207')
    calib['fc2bias'] = CalibrationParam(mpq_config[35], False, -0.1750922054052353, 0.1385071724653244, 'FC2BIAS')
    calib['onnxMatMul_208'] = CalibrationParam(mpq_config[36], False, -3.1666038036346436, 2.5026357173919678, 'ONNX__MATMUL_208')
    calib['fc3bias'] = CalibrationParam(mpq_config[37], False, -0.10188056528568268, 0.0899151861667633, 'FC3BIAS')
    calib['onnxMatMul_209'] = CalibrationParam(mpq_config[38], False, -1.300571084022522, 1.928941249847412, 'ONNX__MATMUL_209')
    calib['fc4bias'] = CalibrationParam(mpq_config[39], False, -0.10699586570262909, 0.04597663879394531, 'FC4BIAS')
    calib['gru1_b_'] = CalibrationParam(mpq_config[40], False, -0.004922409541904926, 0.004103424027562141, 'GRU1_B__')
    calib['gru1_b'] = CalibrationParam(mpq_config[41], False, -0.07475128024816513, 0.14630259573459625, 'GRU1_B_')
    calib['gru1_c_'] = CalibrationParam(mpq_config[42], False, -1.5660111904144287, 1.0454494953155518, 'GRU1_C__')
    calib['gru1_c'] = CalibrationParam(mpq_config[43], False, -1.9779117107391357, 1.3700535297393799, 'GRU1_C_')
    calib['gru1_d_'] = CalibrationParam(mpq_config[44], False, -0.008429044857621193, 0.006403823848813772, 'GRU1_D__')
    calib['gru1_d'] = CalibrationParam(mpq_config[45], False, -0.5529640316963196, 0.41507571935653687, 'GRU1_D_')
    calib['gru1_e_'] = CalibrationParam(mpq_config[46], False, -1.3637111186981201, 1.018247127532959, 'GRU1_E__')
    calib['gru1_e'] = CalibrationParam(mpq_config[47], False, -1.8583884239196777, 1.1587692499160767, 'GRU1_E_')
    calib['gru1_f_'] = CalibrationParam(mpq_config[48], False, -0.005411399528384209, 0.0061536869034171104, 'GRU1_F__')
    calib['gru1_f'] = CalibrationParam(mpq_config[49], False, -0.7833794355392456, 0.8978855013847351, 'GRU1_F_')
    calib['gru1_r_'] = CalibrationParam(mpq_config[50], False, -0.6226556301116943, 0.9800534844398499, 'GRU1_R__')
    calib['gru1_r'] = CalibrationParam(mpq_config[51], False, 0.3491777181625366, 0.7271188497543335, 'GRU1_R_')
    calib['gru1_z_'] = CalibrationParam(mpq_config[52], False, -2.392332077026367, 1.7851293087005615, 'GRU1_Z__')
    calib['gru1_z'] = CalibrationParam(mpq_config[53], False, 0.08375928550958633, 0.856329083442688, 'GRU1_Z_')
    calib['gru1_n1'] = CalibrationParam(mpq_config[54], False, -0.5516456365585327, 0.4556601643562317, 'GRU1_N1_')
    calib['gru1_n2'] = CalibrationParam(mpq_config[55], False, -2.4100341796875, 1.1423156261444092, 'GRU1_N2_')
    calib['gru1_n'] = CalibrationParam(mpq_config[56], False, -0.983996570110321, 0.8151924014091492, 'GRU1_N_')
    calib['gru1_hn1'] = CalibrationParam(mpq_config[57], False, 0.143670916557312, 0.9162406921386719, 'GRU1_HN1_')
    calib['gru1_hn2'] = CalibrationParam(mpq_config[58], False, -0.9015777111053467, 0.45391541719436646, 'GRU1_HN2_')
    calib['gru1_hn3'] = CalibrationParam(mpq_config[59], False, -0.0016954239690676332, 0.0016671211924403906, 'GRU1_HN3_')
    calib['rnn1GRU'] = CalibrationParam(mpq_config[60], False, -0.9016271829605103, 0.4546271562576294, 'RNN1GRU_')
    calib['gru2_a_'] = CalibrationParam(mpq_config[61], False, -0.681461751461029, 0.9113116264343262, 'GRU2_A__')
    calib['gru2_a'] = CalibrationParam(mpq_config[62], False, -0.7149235010147095, 0.908741295337677, 'GRU2_A_')
    calib['gru2_b_'] = CalibrationParam(mpq_config[63], False, -0.005148397758603096, 0.011014967225492, 'GRU2_B__')
    calib['gru2_b'] = CalibrationParam(mpq_config[64], False, -0.09436552226543427, 0.07623167335987091, 'GRU2_B_')
    calib['gru2_c_'] = CalibrationParam(mpq_config[65], False, -1.2279887199401855, 1.1102137565612793, 'GRU2_C__')
    calib['gru2_c'] = CalibrationParam(mpq_config[66], False, -1.6760436296463013, 1.200895071029663, 'GRU2_C_')
    calib['gru2_d_'] = CalibrationParam(mpq_config[67], False, -0.008141756057739258, 0.00894666276872158, 'GRU2_D__')
    calib['gru2_d'] = CalibrationParam(mpq_config[68], False, -0.43875735998153687, 0.1262134611606598, 'GRU2_D_')
    calib['gru2_e_'] = CalibrationParam(mpq_config[69], False, -0.6732944250106812, 0.6314664483070374, 'GRU2_E__')
    calib['gru2_e'] = CalibrationParam(mpq_config[70], False, -0.9132042527198792, 0.7054852247238159, 'GRU2_E_')
    calib['gru2_f_'] = CalibrationParam(mpq_config[71], False, -0.0040821353904902935, 0.005842617247253656, 'GRU2_F__')
    calib['gru2_f'] = CalibrationParam(mpq_config[72], False, -0.17201536893844604, 0.19645990431308746, 'GRU2_F_')
    calib['gru2_r_'] = CalibrationParam(mpq_config[73], False, -0.7412133812904358, 0.8698829412460327, 'GRU2_R__')
    calib['gru2_r'] = CalibrationParam(mpq_config[74], False, 0.3227388560771942, 0.7047213315963745, 'GRU2_R_')
    calib['gru2_z_'] = CalibrationParam(mpq_config[75], False, -2.1148009300231934, 1.2168352603912354, 'GRU2_Z__')
    calib['gru2_z'] = CalibrationParam(mpq_config[76], False, 0.10766655951738358, 0.771506130695343, 'GRU2_Z_')
    calib['gru2_n1'] = CalibrationParam(mpq_config[77], False, -0.09494832903146744, 0.07928616553544998, 'GRU2_N1_')
    calib['gru2_n2'] = CalibrationParam(mpq_config[78], False, -1.0081526041030884, 0.7076360583305359, 'GRU2_N2_')
    calib['gru2_n'] = CalibrationParam(mpq_config[79], False, -0.7649968266487122, 0.6091923117637634, 'GRU2_N_')
    calib['gru2_hn1'] = CalibrationParam(mpq_config[80], False, 0.22849386930465698, 0.892333447933197, 'GRU2_HN1_')
    calib['gru2_hn2'] = CalibrationParam(mpq_config[81], False, -0.6826322674751282, 0.35149604082107544, 'GRU2_HN2_')
    calib['gru2_hn3'] = CalibrationParam(mpq_config[82], False, -0.0016349913785234094, 0.0018691568402573466, 'GRU2_HN3_')
    calib['rnn2GRU'] = CalibrationParam(mpq_config[83], False, -0.6825807690620422, 0.35169717669487, 'RNN2GRU_')
    calib['fc2MatMul'] = CalibrationParam(mpq_config[84], False, -0.769360363483429, 0.43505313992500305, 'FC2_MATMUL_')
    calib['fc2Add'] = CalibrationParam(mpq_config[85], False, -0.9315677881240845, 0.5329311490058899, 'FC2_ADD_')
    calib['relu'] = CalibrationParam(mpq_config[86], False, 0.0, 0.5329311490058899, 'RELU_')
    calib['fc3MatMul'] = CalibrationParam(mpq_config[87], False, -2.872364044189453, 0.898455798625946, 'FC3_MATMUL_')
    calib['fc3Add'] = CalibrationParam(mpq_config[88], False, -2.890881061553955, 0.8957681655883789, 'FC3_ADD_')
    calib['relu_1'] = CalibrationParam(mpq_config[89], False, 0.0, 0.8957681655883789, 'RELU_1_')
    calib['fc4MatMul'] = CalibrationParam(mpq_config[90], False, -2.5626792907714844, -0.9808969497680664, 'FC4_MATMUL_')
    calib['fc4Add'] = CalibrationParam(mpq_config[91], False, -2.569243907928467, -0.9619374871253967, 'FC4_ADD_')
    calib['sigmoid'] = CalibrationParam(mpq_config[92], False, 0.07114425301551819, 0.2764904499053955, 'SIGMOID_')
    return calib