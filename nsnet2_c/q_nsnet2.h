#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float* data_fc1_bias;
static int size_fc1_bias;
static float* data_fc2_bias;
static int size_fc2_bias;
static float* data_fc3_bias;
static int size_fc3_bias;
static float* data_fc4_bias;
static int size_fc4_bias;
static float* data_onnx__GRU_184;
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