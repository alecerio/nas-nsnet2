#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define WRITE_TENSOR_TO_TXT_FILE(array,name,size) \
{ \
FILE *file = fopen(name, "w"); \
for (int i = 0; i < size; i++) { \
    if (i == size - 1) { \
        fprintf(file, "%d", array[i]); \
    } else { \
        fprintf(file, "%d,", array[i]); \
    } \
} \
fclose(file); \
}

#define PRINT_TENSOR(tensor,start,end,type,sep) \
for(int i=start; i<end; i++) { \
    printf(type, tensor[i]); \
} \
printf(sep);

#define PRINT_TENSOR_SUM(tensor,start,end,type,print) \
{ \
type sum = 0; \
for(int i=start; i<end; i++) { \
    sum += tensor[i]; \
} \
printf(print, sum); \
}

#define PRINT_DEBUG_INFO(x, start, end, ssum, esum, type, print, sep) \
PRINT_TENSOR(x, start, end, print, sep) \
PRINT_TENSOR_SUM(x, ssum, esum, type, print) \
printf("\n");

#define CLIP(x, nbits) \
if(x < 0) \
    x = 0; \
else if(x > UINT##nbits##_MAX) \
    x = UINT##nbits##_MAX;

#define QUANTIZE(tensor_fp32,tensor_i,S,Z,size,nbits) \
for(int i=0; i<size; i++) { \
    int64_t acc = (int64_t)(floor(tensor_fp32[i] / S)) + (int64_t)Z; \
    CLIP(acc, nbits) \
    tensor_i[i] = acc; \
}

#define DEQUANTIZE(tensor_q,tensor_f,S,Z,size) \
for(int i=0; i<size; i++) { \
    tensor_f[i] = (float)((tensor_q[i] - Z) * S); \
}

#define MATMUL(rows,cols,matrix,vector,result) \
for (int i = 0; i < rows; i++) { \
    result[i] = 0.0; \
    for (int j = 0; j < cols; j++) { \
        result[i] += matrix[i*cols+j] * vector[j]; \
    } \
}

#define NCAST_ROUND(X) \
(X >= 0) ? (int64_t)(X + 0.5) : (int64_t)(X - 0.5)

#define QMATMUL(rows,cols,matrix,vector,result,Sm,Sv,Sr,Zm,Zv,Zr,res_type) \
{ \
double S = (Sm * Sv) / Sr; \
for (int i = 0; i < rows; i++) { \
    int64_t acc = 0; \
    for (int j = 0; j < cols; j++) { \
        acc += ((int64_t)matrix[i*cols+j]-(int64_t)Zm) * ((int64_t)vector[j]-(int64_t)Zv); \
    } \
    result[i] = (res_type)CLIP((NCAST_ROUND(S * acc + Zr))); \
} \
}

#define QADD(size,A,B,R,Sa,Sb,Sr,Za,Zb,Zr) \
{ \
float Sar = Sa / Sr; \
float Sbr = Sb / Sr; \
for(int i=0; i<size; i++) { \
    R[i] = NCAST_ROUND(Sar * (A[i] - Za) + Sbr * (B[i] - Zb) + Zr); \
} \
}

#define SIGMOID_OP(input,output,Si,Zi,So,Zo,size,temp_sigmoid_x,temp_sigmoid_y) \
DEQUANTIZE(input, temp_sigmoid_x, Si, Zi, size) \
SIGMOID(temp_sigmoid_x, temp_sigmoid_y, size) \
QUANTIZE(temp_sigmoid_y,output,So, Zo, size)

#define SIGMOID(tensor_in,tensor_out,size) \
for(int i=0; i<size; i++) { \
    tensor_out[i] = 1.0f / (1.0f + exp(-tensor_in[i])); \
}

#define TANH_OP(input,output,Si,Zi,So,Zo,size,temp_sigmoid_x,temp_sigmoid_y) \
DEQUANTIZE(input, temp_sigmoid_x, Si, Zi, size) \
TANH(temp_sigmoid_x, temp_sigmoid_y, size) \
QUANTIZE(temp_sigmoid_y,output,So, Zo, size)

#define TANH(tensor_in,tensor_out,size) \
for(int i=0; i<size; i++) { \
    float ex = exp(tensor_in[i]); \
    float emx = exp(-tensor_in[i]); \
    tensor_out[i] = (ex - emx) / (ex + emx); \
}

#define QMUL(A,B,C,Sa,Sb,Sc,Za,Zb,Zc,size) \
{ \
float S = (Sa * Sb) / Sc; \
for(int i=0; i<size; i++) { \
    C[i] = NCAST_ROUND(S * (A[i] - Za) * (B[i] - Zb) + Zc); \
} \
}

#define QUANTIZE_ONE_MINUS_X(x,y,Sx,Sy,Zx,Zy,size) \
{ \
float S1y = S_ones / Sy; \
float Sxy = Sx / Sy; \
float alpha = S1y * (q_ones - Z_ones); \
for(int i=0; i<size; i++) { \
    y[i] = NCAST_ROUND(alpha - Sxy * (x[i] - Zx) + Zy); \
} \
}

#define QRELU(x, y, Sx, Sy, Zx, Zy, size, temp_relu) \
DEQUANTIZE(x, temp_relu, Sx, Zx, size) \
for(int i=0; i<size; i++) { \
    if(temp_relu[i] < 0) { \
        temp_relu[i] = 0; \
    } \
} \
QUANTIZE(temp_relu, y, Sy, Zy, size)

#define TRANSPOSE(rows,cols,matrix,type) \
{ \
type* transposed = (type*) malloc(sizeof(type)*rows*cols); \
for (int i = 0; i < rows; i++) { \
    for (int j = 0; j < cols; j++) { \
        transposed[j*rows+i] = matrix[i*cols+j]; \
    } \
} \
for (int i = 0; i < rows; i++) { \
    for (int j = 0; j < cols; j++) { \
        matrix[i*cols+j] = transposed[i*cols+j]; \
    } \
} \
free(transposed); \
}

#define X_NBITS 8
#define X_TYPE uint8_t
#define X_S (1.8539607843137257e-05)
#define X_Z (135)

#define FC1_BIAS_NBITS 8
#define FC1_BIAS_TYPE uint8_t
#define FC1_BIAS_S (0.0039392154590756285)
#define FC1_BIAS_Z (124)

#define FC2_BIAS_NBITS 8
#define FC2_BIAS_TYPE uint8_t
#define FC2_BIAS_S (0.001229801481845332)
#define FC2_BIAS_Z (142)

#define WIZ_1_NBITS 8
#define WIZ_1_TYPE uint8_t
#define WIZ_1_S (0.003508239517024919)
#define WIZ_1_Z (123)

#define WIR_1_NBITS 8
#define WIR_1_TYPE uint8_t
#define WIR_1_S (0.0024938253795399384)
#define WIR_1_Z (138)

#define WIN_1_NBITS 8
#define WIN_1_TYPE uint8_t
#define WIN_1_S (0.0028225932635513006)
#define WIN_1_Z (115)

#define WHZ_1_NBITS 8
#define WHZ_1_TYPE uint8_t
#define WHZ_1_S (0.01395724287220076)
#define WHZ_1_Z (132)

#define WHR_1_NBITS 8
#define WHR_1_TYPE uint8_t
#define WHR_1_S (0.00857841688043931)
#define WHR_1_Z (135)

#define WHN_1_NBITS 8
#define WHN_1_TYPE uint8_t
#define WHN_1_S (0.006779337864296109)
#define WHN_1_Z (114)

#define BIZ_1_NBITS 8
#define BIZ_1_TYPE uint8_t
#define BIZ_1_S (0.0034234637138890285)
#define BIZ_1_Z (148)

#define BIR_1_NBITS 8
#define BIR_1_TYPE uint8_t
#define BIR_1_S (0.0011188726506981194)
#define BIR_1_Z (71)

#define BIN_1_NBITS 8
#define BIN_1_TYPE uint8_t
#define BIN_1_S (0.0028760029989130355)
#define BIN_1_Z (193)

#define BHZ_1_NBITS 8
#define BHZ_1_TYPE uint8_t
#define BHZ_1_S (0.0037201133428835403)
#define BHZ_1_Z (143)

#define BHR_1_NBITS 8
#define BHR_1_TYPE uint8_t
#define BHR_1_S (0.0008824584238669452)
#define BHR_1_Z (87)

#define BHN_1_NBITS 8
#define BHN_1_TYPE uint8_t
#define BHN_1_S (0.006602613364948946)
#define BHN_1_Z (119)

#define WIZ_2_TYPE uint8_t
#define WIZ_2_S (0.007259108973484413)
#define WIZ_2_Z (125)

#define WIR_2_TYPE uint8_t
#define WIR_2_S (0.006370335700465184)
#define WIR_2_Z (150)

#define WIN_2_TYPE uint8_t
#define WIN_2_S (0.003756115950790106)
#define WIN_2_Z (126)

#define WHZ_2_TYPE uint8_t
#define WHZ_2_S (0.01018969685423608)
#define WHZ_2_Z (128)

#define WHR_2_TYPE uint8_t
#define WHR_2_S (0.007609648564282585)
#define WHR_2_Z (109)

#define WHN_2_TYPE uint8_t
#define WHN_2_S (0.00785202980041504)
#define WHN_2_Z (122)

#define BIZ_2_TYPE uint8_t
#define BIZ_2_S (0.0023688636574090696)
#define BIZ_2_Z (189)

#define BIR_2_TYPE uint8_t
#define BIR_2_S (0.0007888194392709171)
#define BIR_2_Z (111)

#define BIN_2_TYPE uint8_t
#define BIN_2_S (0.0014127153976290835)
#define BIN_2_Z (170)

#define BHZ_2_TYPE uint8_t
#define BHZ_2_S (0.002213522850298414)
#define BHZ_2_Z (198)

#define BHR_2_TYPE uint8_t
#define BHR_2_S (0.0006787231155470306)
#define BHR_2_Z (142)

#define BHN_2_TYPE uint8_t
#define BHN_2_S (0.0014487537683225147)
#define BHN_2_Z (119)

#define ONNX__MATMUL_166_NBITS 8
#define ONNX__MATMUL_166_TYPE uint8_t
#define ONNX__MATMUL_166_S (0.0016850753157746558)
#define ONNX__MATMUL_166_Z (131)

#define ONNX__MATMUL_207_TYPE uint8_t
#define ONNX__MATMUL_207_S (0.009898108594557819)
#define ONNX__MATMUL_207_Z (138)

#define ONNX__MATMUL_208_TYPE uint32_t
#define ONNX__MATMUL_208_S (1.3199726870159115e-09)
#define ONNX__MATMUL_208_Z (2398991915)

#define ONNX__MATMUL_209_TYPE uint32_t
#define ONNX__MATMUL_209_S (7.519294355581197e-10)
#define ONNX__MATMUL_209_Z (1729645127)

#define FC3_BIAS_NBITS 8
#define FC3_BIAS_TYPE uint8_t
#define FC3_BIAS_S (0.000752140201774298)
#define FC3_BIAS_Z (135)

#define FC4_BIAS_NBITS 8
#define FC4_BIAS_TYPE uint8_t
#define FC4_BIAS_S (0.0005998921744963702)
#define FC4_BIAS_Z (178)

#define H1_TYPE uint8_t
#define H1_S (2.149590656307398e-05)
#define H1_Z (123)

#define H2_TYPE uint8_t
#define H2_S (2.402609725501023e-05)
#define H2_Z (129)

#define FC1MATMUL_TYPE uint8_t
#define FC1MATMUL_S (1.8245977529411765e-05)
#define FC1MATMUL_Z (160)

#define FC1ADD_TYPE uint8_t
#define FC1ADD_S (0.00394488257520339)
#define FC1ADD_Z (124)

#define GRU1_A__TYPE uint8_t
#define GRU1_A__S (0.005531594332526712)
#define GRU1_A__Z (116)

#define GRU1_A_TYPE uint8_t
#define GRU1_A_S (0.00585017671772078)
#define GRU1_A_Z (103)

#define GRU1_B__TYPE uint8_t
#define GRU1_B__S (3.5395425762615955e-05)
#define GRU1_B__Z (139)

#define GRU1_B_TYPE uint8_t
#define GRU1_B_S (0.0008668779450304368)
#define GRU1_B_Z (86)

#define GRU1_C__TYPE uint8_t
#define GRU1_C__S (0.010241022296980316)
#define GRU1_C__Z (153)

#define GRU1_C_TYPE uint8_t
#define GRU1_C_S (0.013129275452856925)
#define GRU1_C_Z (151)

#define GRU1_D__TYPE uint8_t
#define GRU1_D__S (5.8168112574254764e-05)
#define GRU1_D__Z (145)

#define GRU1_D_TYPE uint8_t
#define GRU1_D_S (0.003796234317854339)
#define GRU1_D_Z (146)

#define GRU1_E__TYPE uint8_t
#define GRU1_E__S (0.009341012730317958)
#define GRU1_E__Z (146)

#define GRU1_E_TYPE uint8_t
#define GRU1_E_S (0.011831990877787272)
#define GRU1_E_Z (157)

#define GRU1_F__TYPE uint8_t
#define GRU1_F__S (4.5353280124711056e-05)
#define GRU1_F__Z (119)

#define GRU1_F_TYPE uint8_t
#define GRU1_F_S (0.006593195831074435)
#define GRU1_F_Z (119)

#define GRU1_R__TYPE uint8_t
#define GRU1_R__S (0.0062851337825550755)
#define GRU1_R__Z (99)

#define GRU1_R_TYPE uint8_t
#define GRU1_R_S (0.0014821220846737131)
#define GRU1_R_Z (-236)

#define GRU1_Z__TYPE uint8_t
#define GRU1_Z__S (0.016382201512654623)
#define GRU1_Z__Z (146)

#define GRU1_Z_TYPE uint8_t
#define GRU1_Z_S (0.0030296854820905947)
#define GRU1_Z_Z (-28)

#define GRU1_N1_TYPE uint8_t
#define GRU1_N1_S (0.003950218827116723)
#define GRU1_N1_Z (140)

#define GRU1_N2_TYPE uint8_t
#define GRU1_N2_S (0.013930783552281997)
#define GRU1_N2_Z (173)

#define GRU1_N_TYPE uint8_t
#define GRU1_N_S (0.00705564302556655)
#define GRU1_N_Z (139)

#define GRU1_HN1_TYPE uint8_t
#define GRU1_HN1_S (0.0030296853944367054)
#define GRU1_HN1_Z (-47)

#define GRU1_HN2_TYPE uint8_t
#define GRU1_HN2_S (0.005315659326665541)
#define GRU1_HN2_Z (170)

#define GRU1_HN3_TYPE uint8_t
#define GRU1_HN3_S (1.3186451613756956e-05)
#define GRU1_HN3_Z (129)

#define RNN1GRU_TYPE uint8_t
#define RNN1GRU_S (0.0053186444675221165)
#define RNN1GRU_Z (170)

#define GRU2_A__TYPE uint8_t
#define GRU2_A__S (0.00624617010939355)
#define GRU2_A__Z (109)

#define GRU2_A_TYPE uint8_t
#define GRU2_A_S (0.006367312926872104)
#define GRU2_A_Z (112)

#define GRU2_B__TYPE uint8_t
#define GRU2_B__S (6.338574503566705e-05)
#define GRU2_B__Z (81)

#define GRU2_B_TYPE uint8_t
#define GRU2_B_S (0.0006690086102953144)
#define GRU2_B_Z (141)

#define GRU2_C__TYPE uint8_t
#define GRU2_C__S (0.009169421476476333)
#define GRU2_C__Z (134)

#define GRU2_C_TYPE uint8_t
#define GRU2_C_S (0.01128211255167045)
#define GRU2_C_Z (149)

#define GRU2_D__TYPE uint8_t
#define GRU2_D__S (6.701340716259153e-05)
#define GRU2_D__Z (121)

#define GRU2_D_TYPE uint8_t
#define GRU2_D_S (0.0022155718476164574)
#define GRU2_D_Z (198)

#define GRU2_E__TYPE uint8_t
#define GRU2_E__S (0.005116709307128308)
#define GRU2_E__Z (132)

#define GRU2_E_TYPE uint8_t
#define GRU2_E_S (0.006347801872328216)
#define GRU2_E_Z (144)

#define GRU2_F__TYPE uint8_t
#define GRU2_F__S (3.892059857938804e-05)
#define GRU2_F__Z (105)

#define GRU2_F_TYPE uint8_t
#define GRU2_F_S (0.0014450010715746412)
#define GRU2_F_Z (119)

#define GRU2_R__TYPE uint8_t
#define GRU2_R__S (0.006318024794260661)
#define GRU2_R__Z (117)

#define GRU2_R_TYPE uint8_t
#define GRU2_R_S (0.0014979704922320797)
#define GRU2_R_Z (-215)

#define GRU2_Z__TYPE uint8_t
#define GRU2_Z__S (0.013065239962409525)
#define GRU2_Z__Z (162)

#define GRU2_Z_TYPE uint8_t
#define GRU2_Z_S (0.002603292435991998)
#define GRU2_Z_Z (-41)

#define GRU2_N1_TYPE uint8_t
#define GRU2_N1_S (0.0006832725277134017)
#define GRU2_N1_Z (139)

#define GRU2_N2_TYPE uint8_t
#define GRU2_N2_S (0.006728582989935781)
#define GRU2_N2_Z (150)

#define GRU2_N_TYPE uint8_t
#define GRU2_N_S (0.005388977013382258)
#define GRU2_N_Z (142)

#define GRU2_HN1_TYPE uint8_t
#define GRU2_HN1_S (0.002603292465209961)
#define GRU2_HN1_Z (-88)

#define GRU2_HN2_TYPE uint8_t
#define GRU2_HN2_S (0.004055405130573348)
#define GRU2_HN2_Z (168)

#define GRU2_HN3_TYPE uint8_t
#define GRU2_HN3_S (1.3741757720708847e-05)
#define GRU2_HN3_Z (119)

#define RNN2GRU_TYPE uint8_t
#define RNN2GRU_S (0.004055991944144754)
#define RNN2GRU_Z (168)

#define FC2MATMUL_TYPE uint8_t
#define FC2MATMUL_S (0.004723190209444832)
#define FC2MATMUL_Z (163)

#define FC2ADD_TYPE uint8_t
#define FC2ADD_S (0.005743133086784214)
#define FC2ADD_Z (162)

#define RELU_TYPE uint8_t
#define RELU_S (0.0020899260745329017)
#define RELU_Z (0)

#define FC3MATMUL_TYPE uint8_t
#define FC3MATMUL_S (0.014787528795354507)
#define FC3MATMUL_Z (194)

static X_TYPE* data_x_q;
static int size_x = 257;

static H1_TYPE* data_h1_q;
static int size_h1 = 400;

static H2_TYPE* data_h2_q;
static int size_h2 = 400;

static float* data_fc1_bias;
static FC1_BIAS_TYPE* data_fc1_bias_q;
static int size_fc1_bias;

static float* data_fc2_bias;
static FC2_BIAS_TYPE* data_fc2_bias_q;
static int size_fc2_bias;

static float* data_fc3_bias;
static FC3_BIAS_TYPE* data_fc3_bias_q;
static int size_fc3_bias;

static float* data_fc4_bias;
static FC4_BIAS_TYPE* data_fc4_bias_q;
static int size_fc4_bias;

static float* data_onnx__GRU_184;
static float* data_Wiz_1;
static float* data_Wir_1;
static float* data_Win_1;
static WIZ_1_TYPE* data_Wiz_1_q;
static WIR_1_TYPE* data_Wir_1_q;
static WIN_1_TYPE* data_Win_1_q;
static int size_onnx__GRU_184;

static float* data_onnx__GRU_185;
static float* data_Whz_1;
static float* data_Whr_1;
static float* data_Whn_1;
static WHZ_1_TYPE* data_Whz_1_q;
static WHR_1_TYPE* data_Whr_1_q;
static WHN_1_TYPE* data_Whn_1_q;
static int size_onnx__GRU_185;

static float* data_onnx__GRU_186;
static float* data_biz_1;
static float* data_bir_1;
static float* data_bin_1;
static float* data_bhz_1;
static float* data_bhr_1;
static float* data_bhn_1;
static BIZ_1_TYPE* data_biz_1_q;
static BIR_1_TYPE* data_bir_1_q;
static BIN_1_TYPE* data_bin_1_q;
static BHZ_1_TYPE* data_bhz_1_q;
static BHR_1_TYPE* data_bhr_1_q;
static BHN_1_TYPE* data_bhn_1_q;
static int size_onnx__GRU_186;

static float* data_Wiz_2;
static float* data_onnx__GRU_204;
static float* data_Wir_2;
static float* data_Win_2;
static WIZ_2_TYPE* data_Wiz_2_q;
static WIR_2_TYPE* data_Wir_2_q;
static WIN_2_TYPE* data_Win_2_q;
static int size_onnx__GRU_204;

static float* data_onnx__GRU_205;
static float* data_Whz_2;
static float* data_Whr_2;
static float* data_Whn_2;
static WHZ_2_TYPE* data_Whz_2_q;
static WHR_2_TYPE* data_Whr_2_q;
static WHR_2_TYPE* data_Whn_2_q;
static int size_onnx__GRU_205;

static float* data_onnx__GRU_206;
static float* data_biz_2;
static float* data_bir_2;
static float* data_bin_2;
static float* data_bhz_2;
static float* data_bhr_2;
static float* data_bhn_2;
static BIZ_2_TYPE* data_biz_2_q;
static BIR_2_TYPE* data_bir_2_q;
static BIN_2_TYPE* data_bin_2_q;
static BHZ_2_TYPE* data_bhz_2_q;
static BHR_2_TYPE* data_bhr_2_q;
static BHN_2_TYPE* data_bhn_2_q;
static int size_onnx__GRU_206;

static float* data_onnx__MatMul_166;
static ONNX__MATMUL_166_TYPE* data_onnx__MatMul_166_q;
static int size_onnx__MatMul_166;

static float* data_onnx__MatMul_207;
static ONNX__MATMUL_207_TYPE* data_onnx__MatMul_207_q;
static int size_onnx__MatMul_207;

static float* data_onnx__MatMul_208;
static ONNX__MATMUL_208_TYPE* data_onnx__MatMul_208_q;
static int size_onnx__MatMul_208;

static float* data_onnx__MatMul_209;
static ONNX__MATMUL_209_TYPE* data_onnx__MatMul_209_q;
static int size_onnx__MatMul_209;

static FC1MATMUL_TYPE* data_fc1MatMul_q;
static int size_fc1MatMul = 400;

static FC1ADD_TYPE* data_fc1Add_q;
static int size_fc1Add = 400;

static GRU1_A__TYPE* data_gru1_a__q;
static int size_gru1_a_ = 400;

static GRU1_A_TYPE* data_gru1_a_q;
static int size_gru1_a = 400;

static GRU1_B__TYPE* data_gru1_b__q;
static int size_gru1_b_ = 400;

static GRU1_B_TYPE* data_gru1_b_q;
static int size_gru1_b = 400;

static GRU1_C__TYPE* data_gru1_c__q;
static int size_gru1_c_ = 400;

static GRU1_C_TYPE* data_gru1_c_q;
static int size_gru1_c = 400;

static GRU1_D__TYPE* data_gru1_d__q;
static int size_gru1_d_ = 400;

static GRU1_D_TYPE* data_gru1_d_q;
static int size_gru1_d = 400;

static GRU1_E__TYPE* data_gru1_e__q;
static int size_gru1_e_ = 400;

static GRU1_E_TYPE* data_gru1_e_q;
static int size_gru1_e = 400;

static GRU1_F__TYPE* data_gru1_f__q;
static int size_gru1_f_ = 400;

static GRU1_F_TYPE* data_gru1_f_q;
static int size_gru1_f = 400;

static GRU1_R__TYPE* data_gru1_r__q;
static int size_gru1_r_ = 400;

static GRU1_R_TYPE* data_gru1_r_q;
static int size_gru1_r = 400;

static GRU1_Z__TYPE* data_gru1_z__q;
static int size_gru1_z_ = 400;

static GRU1_Z_TYPE* data_gru1_z_q;
static int size_gru1_z = 400;

static GRU1_N1_TYPE* data_gru1_n1_q;
static int size_gru1_n1 = 400;

static GRU1_N2_TYPE* data_gru1_n2_q;
static int size_gru1_n2 = 400;

static GRU1_N_TYPE* data_gru1_n_q;
static int size_gru1_n = 400;

static GRU1_HN1_TYPE* data_gru1_hn1_q;
static int size_gru1_hn1 = 400;

static GRU1_HN2_TYPE* data_gru1_hn2_q;
static int size_gru1_hn2 = 400;

static GRU1_HN3_TYPE* data_gru1_hn3_q;
static int size_gru1_hn3 = 400;

static RNN1GRU_TYPE* data_rnn1gru_q;
static int size_rnn1gru = 400;

static GRU2_A__TYPE* data_gru2_a__q;
static int size_gru2_a_ = 400;

static GRU2_A_TYPE* data_gru2_a_q;
static int size_gru2_a = 400;

static GRU2_B__TYPE* data_gru2_b__q;
static int size_gru2_b_ = 400;

static GRU2_B_TYPE* data_gru2_b_q;
static int size_gru2_b = 400;

static GRU2_C__TYPE* data_gru2_c__q;
static int size_gru2_c_ = 400;

static GRU2_C_TYPE* data_gru2_c_q;
static int size_gru2_c = 400;

static GRU2_D__TYPE* data_gru2_d__q;
static int size_gru2_d_ = 400;

static GRU2_D__TYPE* data_gru2_d_q;
static int size_gru2_d = 400;

static GRU2_E__TYPE* data_gru2_e__q;
static int size_gru2_e_ = 400;

static GRU2_E_TYPE* data_gru2_e_q;
static int size_gru2_e = 400;

static GRU2_F__TYPE* data_gru2_f__q;
static int size_gru2_f_ = 400;

static GRU2_F_TYPE* data_gru2_f_q;
static int size_gru2_f = 400;

static GRU2_R__TYPE* data_gru2_r__q;
static int size_gru2_r_ = 400;

static GRU2_R_TYPE* data_gru2_r_q;
static int size_gru2_r = 400;

static GRU2_Z__TYPE* data_gru2_z__q;
static int size_gru2_z_ = 400;

static GRU2_Z_TYPE* data_gru2_z_q;
static int size_gru2_z = 400;

static GRU2_N1_TYPE* data_gru2_n1_q;
static int size_gru2_n1 = 400;

static GRU2_N2_TYPE* data_gru2_n2_q;
static int size_gru2_n2 = 400;

static GRU2_N_TYPE* data_gru2_n_q;
static int size_gru2_n = 400;

static GRU2_HN1_TYPE* data_gru2_hn1_q;
static int size_gru2_hn1 = 400;

static GRU2_HN2_TYPE* data_gru2_hn2_q;
static int size_gru2_hn2 = 400;

static GRU2_HN3_TYPE* data_gru2_hn3_q;
static int size_gru2_hn3 = 400;

static RNN2GRU_TYPE* data_rnn2gru_q;
static int size_rnn2gru = 400;

static FC2MATMUL_TYPE* data_fc2MatMul_q;
static int size_fc2MatMul = 600;

static FC2ADD_TYPE* data_fc2Add_q;
static int size_fc2Add = 600;

static RELU_TYPE* data_relu_q;
static int size_relu = 600;

static FC3MATMUL_TYPE* data_fc3MatMul_q;
static int size_fc3MatMul = 600;

static float* temp_sigmoid_x;
static float* temp_sigmoid_y;
static float* temp_relu;

static uint8_t q_ones;
static float S_ones = 1.0 / 255.0;
static int32_t Z_ones = 0;

int setup_nsnet2(const char* weights_path);
void free_nsnet2();
void run_nsnet2(float* x, float* h1, float* h2);

static int read_weights(const char* weights_path, const char* weights_name, float** data, int* size);