
#ifndef __QNSNET2__
#define __QNSNET2__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "mpq.h"

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

#define CLIP(x, clip_val) \
if(x < 0) \
    x = 0; \
else if(x > clip_val) \
    x = clip_val;

#define QUANTIZE(tensor_fp32,tensor_i,S,Z,size,clip_val) \
for(int i=0; i<size; i++) { \
    int64_t acc = (int64_t)(floor(tensor_fp32[i] / S)) + (int64_t)Z; \
    CLIP(acc, clip_val) \
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

#define QMATMUL(rows,cols,matrix,vector,result,Sm,Sv,Sr,Zm,Zv,Zr,res_type,clip_val) \
{ \
double S = (Sm * Sv) / Sr; \
for (int i = 0; i < rows; i++) { \
    int64_t acc = 0; \
    for (int j = 0; j < cols; j++) { \
        acc += ((int64_t)matrix[i*cols+j]-(int64_t)Zm) * ((int64_t)vector[j]-(int64_t)Zv); \
    } \
    acc = NCAST_ROUND(S * acc + Zr); \
    CLIP(acc,clip_val); \
    result[i] = (res_type) acc; \
} \
}

#define QADD(size,A,B,R,Sa,Sb,Sr,Za,Zb,Zr,clip_val) \
{ \
int64_t acc; \
float Sar = Sa / Sr; \
float Sbr = Sb / Sr; \
for(int i=0; i<size; i++) { \
    acc = NCAST_ROUND(Sar * (A[i] - Za) + Sbr * (B[i] - Zb) + Zr); \
    CLIP(acc, clip_val) \
    R[i] = acc; \
} \
}

#define SIGMOID_OP(input,output,Si,Zi,So,Zo,size,temp_sigmoid_x,temp_sigmoid_y,clip_val) \
DEQUANTIZE(input, temp_sigmoid_x, Si, Zi, size) \
SIGMOID(temp_sigmoid_x, temp_sigmoid_y, size) \
QUANTIZE(temp_sigmoid_y,output,So, Zo, size,clip_val)

#define SIGMOID(tensor_in,tensor_out,size) \
for(int i=0; i<size; i++) { \
    tensor_out[i] = 1.0f / (1.0f + exp(-tensor_in[i])); \
}

#define TANH_OP(input,output,Si,Zi,So,Zo,size,temp_sigmoid_x,temp_sigmoid_y,clip_val) \
DEQUANTIZE(input, temp_sigmoid_x, Si, Zi, size) \
TANH(temp_sigmoid_x, temp_sigmoid_y, size) \
QUANTIZE(temp_sigmoid_y,output,So, Zo, size,clip_val)

#define TANH(tensor_in,tensor_out,size) \
for(int i=0; i<size; i++) { \
    float ex = exp(tensor_in[i]); \
    float emx = exp(-tensor_in[i]); \
    tensor_out[i] = (ex - emx) / (ex + emx); \
}

#define QMUL(A,B,C,Sa,Sb,Sc,Za,Zb,Zc,size,clip_val) \
{ \
int64_t acc = 0; \
float S = (Sa * Sb) / Sc; \
for(int i=0; i<size; i++) { \
    acc = NCAST_ROUND(S * (A[i] - Za) * (B[i] - Zb) + Zc); \
    CLIP(acc, clip_val) \
    C[i] = acc; \
} \
}

#define QUANTIZE_ONE_MINUS_X(x,y,Sx,Sy,Zx,Zy,size,clip_val) \
{ \
float S1y = S_ones / Sy; \
float Sxy = Sx / Sy; \
float alpha = S1y * (q_ones - Z_ones); \
int64_t acc; \
for(int i=0; i<size; i++) { \
    acc = NCAST_ROUND(alpha - Sxy * (x[i] - Zx) + Zy); \
    CLIP(acc, clip_val) \
    y[i] = acc; \
} \
}

#define QRELU(x,y,Sx,Sy,Zx,Zy,size,temp_relu,clip_val) \
DEQUANTIZE(x, temp_relu, Sx, Zx, size) \
for(int i=0; i<size; i++) { \
    if(temp_relu[i] < 0) { \
        temp_relu[i] = 0; \
    } \
} \
QUANTIZE(temp_relu, y, Sy, Zy, size, clip_val)

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
static WHN_2_TYPE* data_Whn_2_q;
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

static GRU2_D_TYPE* data_gru2_d_q;
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

static FC3ADD_TYPE* data_fc3Add_q;
static int size_fc3Add = 600;

static RELU_1_TYPE* data_relu_1_q;
static int size_relu_1 = 600;

static FC4MATMUL_TYPE* data_fc4MatMul_q;
static int size_fc4MatMul = 257;

static FC4ADD_TYPE* data_fc4Add_q;
static int size_fc4Add = 257;

static SIGMOID_TYPE* data_sigmoid_q;
static int size_sigmoid = 257;

static float* data_output;
static int size_output = 257;

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

#endif // __QNSNET2__