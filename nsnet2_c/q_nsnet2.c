#include "q_nsnet2.h"

int setup_nsnet2(const char* weights_path) {
    int flag;

    // fc1_bias
    flag = read_weights(weights_path, "fc1_bias.npy", &data_fc1_bias, &size_fc1_bias);
    if(flag != 0)
        return -1;
    data_fc1_bias_q = (FC1_BIAS_TYPE*) malloc(sizeof(FC1_BIAS_TYPE) * size_fc1_bias);
    QUANTIZE(data_fc1_bias,data_fc1_bias_q,FC1_BIAS_S,FC1_BIAS_Z,size_fc1_bias)
    free(data_fc1_bias);
    
    // fc2_bias
    flag = read_weights(weights_path, "fc2_bias.npy", &data_fc2_bias, &size_fc2_bias);
    if(flag != 0)
        return -1;
    data_fc2_bias_q = (FC2_BIAS_TYPE*) malloc(sizeof(FC2_BIAS_TYPE) * size_fc2_bias);
    QUANTIZE(data_fc2_bias,data_fc2_bias_q,FC2_BIAS_S,FC2_BIAS_Z,size_fc2_bias)
    free(data_fc2_bias);
    
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
    
    data_Wiz_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wiz_1[i*400+j] = data_onnx__GRU_184[i*400+j];
        }
    }
    data_Wiz_1_q = (WIZ_1_TYPE*) malloc(sizeof(WIZ_1_TYPE)*400*400);
    QUANTIZE(data_Wiz_1, data_Wiz_1_q, WIZ_1_S, WIZ_1_Z, 400*400)
    free(data_Wiz_1);

    data_Wir_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wir_1[i*400+j] = data_onnx__GRU_184[400*400+i*400+j];
        }
    }
    data_Wir_1_q = (WIR_1_TYPE*) malloc(sizeof(WIR_1_TYPE)*400*400);
    QUANTIZE(data_Wir_1, data_Wir_1_q, WIR_1_S, WIR_1_Z, 400*400)
    free(data_Wir_1);

    data_Win_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Win_1[i*400+j] = data_onnx__GRU_184[800*400+i*400+j];
        }
    }
    data_Win_1_q = (WIN_1_TYPE*) malloc(sizeof(WIN_1_TYPE)*400*400);
    QUANTIZE(data_Win_1, data_Win_1_q, WIN_1_S, WIN_1_Z, 400*400)
    free(data_Win_1);
    
    // onnx__GRU_185
    flag = read_weights(weights_path, "onnx__GRU_185.npy", &data_onnx__GRU_185, &size_onnx__GRU_185);
    if(flag != 0)
        return -1;
    
    data_Whz_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whz_1[i*400+j] = data_onnx__GRU_185[i*400+j];
        }
    }
    data_Whz_1_q = (WHZ_1_TYPE*) malloc(sizeof(WHZ_1_TYPE)*400*400);
    QUANTIZE(data_Whz_1, data_Whz_1_q, WHZ_1_S, WHZ_1_Z, 400*400)
    free(data_Whz_1);

    data_Whr_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whr_1[i*400+j] = data_onnx__GRU_185[400*400+i*400+j];
        }
    }
    data_Whr_1_q = (WHR_1_TYPE*) malloc(sizeof(WHR_1_TYPE)*400*400);
    QUANTIZE(data_Whr_1, data_Whr_1_q, WHR_1_S, WHR_1_Z, 400*400)
    free(data_Whr_1);

    data_Whn_1 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whn_1[i*400+j] = data_onnx__GRU_185[800*400+i*400+j];
        }
    }
    data_Whn_1_q = (WHN_1_TYPE*) malloc(sizeof(WHN_1_TYPE)*400*400);
    QUANTIZE(data_Whn_1, data_Whn_1_q, WHN_1_S, WHN_1_Z, 400*400)
    free(data_Whn_1);
    
    // onnx__GRU_186
    flag = read_weights(weights_path, "onnx__GRU_186.npy", &data_onnx__GRU_186, &size_onnx__GRU_186);
    if(flag != 0)
        return -1;
    
    data_biz_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_biz_1[i] = data_onnx__GRU_186[i];
    }
    data_biz_1_q = (BIZ_1_TYPE*) malloc(sizeof(BIZ_1_TYPE)*400);
    QUANTIZE(data_biz_1, data_biz_1_q, BIZ_1_S, BIZ_1_Z, 400)
    free(data_biz_1);

    data_bir_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bir_1[i] = data_onnx__GRU_186[400+i];
    }
    data_bir_1_q = (BIR_1_TYPE*) malloc(sizeof(BIR_1_TYPE)*400);
    QUANTIZE(data_bir_1, data_bir_1_q, BIR_1_S, BIR_1_Z, 400)
    free(data_bir_1);

    data_bin_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bin_1[i] = data_onnx__GRU_186[800+i];
    }
    data_bin_1_q = (BIN_1_TYPE*) malloc(sizeof(BIN_1_TYPE)*400);
    QUANTIZE(data_bin_1, data_bin_1_q, BIN_1_S, BIN_1_Z, 400)
    free(data_bin_1);

    data_bhz_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhz_1[i] = data_onnx__GRU_186[1200+i];
    }
    data_bhz_1_q = (BHZ_1_TYPE*) malloc(sizeof(BHZ_1_TYPE)*400);
    QUANTIZE(data_bhz_1, data_bhz_1_q, BHZ_1_S, BHZ_1_Z, 400)
    free(data_bhz_1);

    data_bhr_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhr_1[i] = data_onnx__GRU_186[1600+i];
    }
    data_bhr_1_q = (BHR_1_TYPE*) malloc(sizeof(BHR_1_TYPE)*400);
    QUANTIZE(data_bhr_1, data_bhr_1_q, BHR_1_S, BHR_1_Z, 400)
    free(data_bhr_1);

    data_bhn_1 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhn_1[i] = data_onnx__GRU_186[2000+i];
    }
    data_bhn_1_q = (BHN_1_TYPE*) malloc(sizeof(BHN_1_TYPE)*400);
    QUANTIZE(data_bhn_1, data_bhn_1_q, BHN_1_S, BHN_1_Z, 400)
    free(data_bhn_1);

    // onnx__GRU_204
    flag = read_weights(weights_path, "onnx__GRU_204.npy", &data_onnx__GRU_204, &size_onnx__GRU_204);
    if(flag != 0)
        return -1;
    
    data_Wiz_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wiz_2[i*400+j] = data_onnx__GRU_204[i*400+j];
        }
    }
    data_Wiz_2_q = (WIZ_2_TYPE*) malloc(sizeof(WIZ_2_TYPE)*400*400);
    QUANTIZE(data_Wiz_2, data_Wiz_2_q, WIZ_2_S, WIZ_2_Z, 400*400)
    PRINT_TENSOR(data_Wiz_2_q, 0, 5, "%d ", "\n")
    PRINT_TENSOR(data_Wiz_2_q, 400*400-5, 400*400, "%d ", "\n")
    free(data_Wiz_2);

    data_Wir_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wir_2[i*400+j] = data_onnx__GRU_204[400*400+i*400+j];
        }
    }
    free(data_Wir_2);

    data_Win_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wir_2[i*400+j] = data_onnx__GRU_204[800*400+i*400+j];
        }
    }
    free(data_Win_2);
    
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
    free(data_fc1_bias_q);
    free(data_fc2_bias_q);
    //free(data_fc3_bias);
    //free(data_fc4_bias);
    //free(data_onnx__GRU_184);
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
