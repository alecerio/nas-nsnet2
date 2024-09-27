#include "q_nsnet2.h"

int setup_nsnet2(const char* weights_path) {
    int flag;

    // fc1_bias
    flag = read_weights(weights_path, "fc1_bias.npy", &data_fc1_bias, &size_fc1_bias);
    if(flag != 0)
        return -1;
    
    // fc2_bias
    flag = read_weights(weights_path, "fc2_bias.npy", &data_fc2_bias, &size_fc2_bias);
    if(flag != 0)
        return -1;
    
    // fc3_bias
    flag = read_weights(weights_path, "fc3_bias.npy", &data_fc3_bias, &size_fc3_bias);
    if(flag != 0)
        return -1;
    
    // fc4_bias
    flag = read_weights(weights_path, "fc4_bias.npy", &data_fc4_bias, &size_fc4_bias);
    if(flag != 0)
        return -1;

    // onnx__GRU_184
    flag = read_weights(weights_path, "onnx__GRU_184.npy", &data_onnx__GRU_184, &size_onnx__GRU_184);
    if(flag != 0)
        return -1;
    
    // onnx__GRU_185
    flag = read_weights(weights_path, "onnx__GRU_185.npy", &data_onnx__GRU_185, &size_onnx__GRU_185);
    if(flag != 0)
        return -1;
    
    // onnx__GRU_186
    flag = read_weights(weights_path, "onnx__GRU_186.npy", &data_onnx__GRU_186, &size_onnx__GRU_186);
    if(flag != 0)
        return -1;
    
    // onnx__GRU_204
    flag = read_weights(weights_path, "onnx__GRU_204.npy", &data_onnx__GRU_204, &size_onnx__GRU_204);
    if(flag != 0)
        return -1;
    
    // onnx__GRU_205
    flag = read_weights(weights_path, "onnx__GRU_205.npy", &data_onnx__GRU_205, &size_onnx__GRU_205);
    if(flag != 0)
        return -1;
    
    // onnx__GRU_206
    flag = read_weights(weights_path, "onnx__GRU_206.npy", &data_onnx__GRU_206, &size_onnx__GRU_206);
    if(flag != 0)
        return -1;
    
    // onnx__MatMul_166
    flag = read_weights(weights_path, "onnx__MatMul_166.npy", &data_onnx__MatMul_166, &size_onnx__MatMul_166);
    if(flag != 0)
        return -1;
    
    // onnx__MatMul_207
    flag = read_weights(weights_path, "onnx__MatMul_207.npy", &data_onnx__MatMul_207, &size_onnx__MatMul_207);
    if(flag != 0)
        return -1;
    
    // onnx__MatMul_208
    flag = read_weights(weights_path, "onnx__MatMul_208.npy", &data_onnx__MatMul_208, &size_onnx__MatMul_208);
    if(flag != 0)
        return -1;
    
    // onnx__MatMul_209
    flag = read_weights(weights_path, "onnx__MatMul_209.npy", &data_onnx__MatMul_209, &size_onnx__MatMul_209);
    if(flag != 0)
        return -1;
    
    
    /*printf("size: %d\n", size_onnx__GRU_184);
    for(int i=0; i<size_onnx__MatMul_166; i++) {
        printf("%f ", data_onnx__MatMul_166[i]);
    }*/

    return 0;
}

void free_nsnet2() {
    free(data_fc1_bias);
    free(data_fc2_bias);
    free(data_fc3_bias);
    free(data_fc4_bias);
    free(data_onnx__GRU_184);
}

int read_weights(const char* weights_path, const char* weights_name, float** data, int* size) {
    int len_name = strlen(weights_name);
    int len_path = strlen(weights_path);
    char* weights_path_full = (char*) malloc(sizeof(char) * (len_path + len_name));
    strcpy(weights_path_full, weights_path);
    strcpy(weights_path_full+len_path, weights_name);
    FILE *file = fopen(weights_path_full, "rb");
    if (!file) {
        printf("Error: file cannot be read.\n");
        return -1;
    }
    int flag = read_npy_file(file, data, size);
    /*for(int i=0; i<*size; i++) {
        printf("%d ", i);
        printf("%f ", (*data)[i]);
    }*/
    fclose(file);
    free(weights_path_full);
    if(flag != 0) {
        printf("Error with reading weights\n");
        return -1;
    }
    return 0;
}
