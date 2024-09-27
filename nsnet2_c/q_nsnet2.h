#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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
static int size_onnx__GRU_184;

static float* data_onnx__GRU_185;
static int size_onnx__GRU_185;

static float* data_onnx__GRU_186;
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