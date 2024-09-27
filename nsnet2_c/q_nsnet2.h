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

int setup_nsnet2(const char* weights_path);
void free_nsnet2();

static int read_weights(const char* weights_path, const char* weights_name, float** data, int* size);