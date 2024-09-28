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
static int size_onnx__GRU_186;

static float* data_onnx__GRU_204;
static int size_onnx__GRU_204;

static float* data_onnx__GRU_205;
static int size_onnx__GRU_205;

static float* data_onnx__GRU_206;
static int size_onnx__GRU_206;

static float* data_onnx__MatMul_166;
static int size_onnx__MatMul_166;

static float* data_onnx__MatMul_207;
static int size_onnx__MatMul_207;

static float* data_onnx__MatMul_208;
static int size_onnx__MatMul_208;

static float* data_onnx__MatMul_209;
static int size_onnx__MatMul_209;

int setup_nsnet2(const char* weights_path);
void free_nsnet2();

static int read_weights(const char* weights_path, const char* weights_name, float** data, int* size);