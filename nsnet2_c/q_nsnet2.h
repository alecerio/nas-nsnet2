#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define PRINT_TENSOR(tensor,start,end,type,sep) \
for(int i=start; i<end; i++) { \
    printf(type, tensor[i]); \
} \
printf(sep);

#define QUANTIZE(tensor_fp32,tensor_i,S,Z,size) \
for(int i=0; i<size; i++) { \
    tensor_i[i] = (int32_t)(floor(tensor_fp32[i] / S)) + Z; \
}

#define MATMUL(rows,cols,matrix,vector,result) \
for (int i = 0; i < rows; i++) { \
    result[i] = 0.0; \
    for (int j = 0; j < cols; j++) { \
        result[i] += matrix[i*cols+j] * vector[j]; \
    } \
}

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

#define X_TYPE uint8_t
#define X_S (1.8539607843137257e-05f)
#define X_Z (135)

#define FC1_BIAS_TYPE uint8_t
#define FC1_BIAS_S (0.0039392154590756285)
#define FC1_BIAS_Z (124)

#define FC2_BIAS_TYPE uint8_t
#define FC2_BIAS_S (0.001229801481845332f)
#define FC2_BIAS_Z (142)

#define WIZ_1_TYPE uint8_t
#define WIZ_1_S (0.003508239517024919)
#define WIZ_1_Z (123)

#define WIR_1_TYPE uint8_t
#define WIR_1_S (0.0024938253795399384)
#define WIR_1_Z (138)

#define WIN_1_TYPE uint8_t
#define WIN_1_S (0.0028225932635513006)
#define WIN_1_Z (115)

#define WHZ_1_TYPE uint8_t
#define WHZ_1_S (0.01395724287220076f)
#define WHZ_1_Z (132)

#define WHR_1_TYPE uint8_t
#define WHR_1_S (0.00857841688043931f)
#define WHR_1_Z (135)

#define WHN_1_TYPE uint8_t
#define WHN_1_S (0.006779337864296109f)
#define WHN_1_Z (114)

#define BIZ_1_TYPE uint8_t
#define BIZ_1_S (0.0034234637138890285f)
#define BIZ_1_Z (148)

#define BIR_1_TYPE uint8_t
#define BIR_1_S (0.0011188726506981194)
#define BIR_1_Z (71)

#define BIN_1_TYPE uint8_t
#define BIN_1_S (0.0028760029989130355)
#define BIN_1_Z (193)

#define BHZ_1_TYPE uint8_t
#define BHZ_1_S (0.0037201133428835403)
#define BHZ_1_Z (143)

#define BHR_1_TYPE uint8_t
#define BHR_1_S (0.0008824584238669452)
#define BHR_1_Z (87)

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
#define WHZ_2_S (0.01018969685423608f)
#define WHZ_2_Z (128)

#define WHR_2_TYPE uint8_t
#define WHR_2_S (0.007609648564282585f)
#define WHR_2_Z (109)

#define WHN_2_TYPE uint8_t
#define WHN_2_S (0.00785202980041504f)
#define WHN_2_Z (122)

#define BIZ_2_TYPE uint8_t
#define BIZ_2_S (0.0023688636574090696f)
#define BIZ_2_Z (189)

#define BIR_2_TYPE uint8_t
#define BIR_2_S (0.0007888194392709171f)
#define BIR_2_Z (111)

#define BIN_2_TYPE uint8_t
#define BIN_2_S (0.0014127153976290835f)
#define BIN_2_Z (170)

#define BHZ_2_TYPE uint8_t
#define BHZ_2_S (0.002213522850298414f)
#define BHZ_2_Z (198)

#define BHR_2_TYPE uint8_t
#define BHR_2_S (0.0006787231155470306f)
#define BHR_2_Z (142)

#define BHN_2_TYPE uint8_t
#define BHN_2_S (0.0014487537683225147f)
#define BHN_2_Z (119)

#define ONNX__MATMUL_166_TYPE uint8_t
#define ONNX__MATMUL_166_S (0.0016850753157746558f)
#define ONNX__MATMUL_166_Z (131)

#define ONNX__MATMUL_207_TYPE uint8_t
#define ONNX__MATMUL_207_S (0.009898108594557819f)
#define ONNX__MATMUL_207_Z (138)

static X_TYPE* data_x_q;
static int size_x = 257;

static float* data_fc1_bias;
static FC1_BIAS_TYPE* data_fc1_bias_q;
static int size_fc1_bias;

static float* data_fc2_bias;
static FC2_BIAS_TYPE* data_fc2_bias_q;
static int size_fc2_bias;

static float* data_fc3_bias;
static int size_fc3_bias;

static float* data_fc4_bias;
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

static float* data_onnx__GRU_204;
static float* data_Wiz_2;
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
static int size_onnx__MatMul_208;

static float* data_onnx__MatMul_209;
static int size_onnx__MatMul_209;

static float* data_fc1MatMul;
static int size_fc1MatMul;

int setup_nsnet2(const char* weights_path);
void free_nsnet2();
void run_nsnet2(float* x, float* h1, float* h2);

static int read_weights(const char* weights_path, const char* weights_name, float** data, int* size);