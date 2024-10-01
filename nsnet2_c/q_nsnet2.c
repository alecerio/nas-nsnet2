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

    // data_onnx__MatMul_166
    flag = read_weights(weights_path, "onnx__MatMul_166.npy", &data_onnx__MatMul_166, &size_onnx__MatMul_166);
    if(flag != 0)
        return -1;
    TRANSPOSE(257, 400, data_onnx__MatMul_166, float)
    data_onnx__MatMul_166_q = (ONNX__MATMUL_166_TYPE*) malloc(sizeof(ONNX__MATMUL_166_TYPE) * size_onnx__MatMul_166);
    QUANTIZE(data_onnx__MatMul_166, data_onnx__MatMul_166_q, ONNX__MATMUL_166_S, ONNX__MATMUL_166_Z, size_onnx__MatMul_166)
    free(data_onnx__MatMul_166);
    
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
    data_fc3_bias_q = (FC3_BIAS_TYPE*) malloc(sizeof(FC3_BIAS_TYPE) * size_fc3_bias);
    QUANTIZE(data_fc3_bias,data_fc3_bias_q,FC3_BIAS_S,FC3_BIAS_Z,size_fc3_bias)   
    free(data_fc3_bias);
    
    // fc4_bias
    flag = read_weights(weights_path, "fc4_bias.npy", &data_fc4_bias, &size_fc4_bias);
    if(flag != 0)
        return -1;
    data_fc4_bias_q = (FC4_BIAS_TYPE*) malloc(sizeof(FC4_BIAS_TYPE) * size_fc4_bias);
    QUANTIZE(data_fc4_bias,data_fc4_bias_q,FC4_BIAS_S,FC4_BIAS_Z,size_fc4_bias)   
    free(data_fc4_bias);

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
    free(data_Wiz_2);

    data_Wir_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wir_2[i*400+j] = data_onnx__GRU_204[400*400+i*400+j];
        }
    }
    data_Wir_2_q = (WIR_2_TYPE*) malloc(sizeof(WIR_2_TYPE)*400*400);
    QUANTIZE(data_Wir_2, data_Wir_2_q, WIR_2_S, WIR_2_Z, 400*400)
    free(data_Wir_2);

    data_Win_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Wir_2[i*400+j] = data_onnx__GRU_204[800*400+i*400+j];
        }
    }
    data_Win_2_q = (WIN_2_TYPE*) malloc(sizeof(WIN_2_TYPE)*400*400);
    QUANTIZE(data_Win_2, data_Win_2_q, WIN_2_S, WIN_2_Z, 400*400)
    free(data_Win_2);
    
    // onnx__GRU_205
    flag = read_weights(weights_path, "onnx__GRU_205.npy", &data_onnx__GRU_205, &size_onnx__GRU_205);
    if(flag != 0)
        return -1;
    
    data_Whz_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whz_2[i*400+j] = data_onnx__GRU_205[i*400+j];
        }
    }
    data_Whz_2_q = (WHZ_2_TYPE*) malloc(sizeof(WHZ_2_TYPE)*400*400);
    QUANTIZE(data_Whz_2, data_Whz_2_q, WHZ_2_S, WHZ_2_Z, 400*400)
    free(data_Whz_2);

    data_Whr_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whr_2[i*400+j] = data_onnx__GRU_205[400*400+i*400+j];
        }
    }
    data_Whr_2_q = (WHR_2_TYPE*) malloc(sizeof(WHR_2_TYPE)*400*400);
    QUANTIZE(data_Whr_2, data_Whr_2_q, WHR_2_S, WHR_2_Z, 400*400)
    free(data_Whr_2);

    data_Whn_2 = (float*) malloc(sizeof(float)*400*400);
    for(int i=0; i<400; i++) {
        for(int j=0; j<400; j++) {
            data_Whn_2[i*400+j] = data_onnx__GRU_205[800*400+i*400+j];
        }
    }
    data_Whn_2_q = (WHN_2_TYPE*) malloc(sizeof(WHN_2_TYPE)*400*400);
    QUANTIZE(data_Whn_2, data_Whn_2_q, WHN_2_S, WHN_2_Z, 400*400)
    free(data_Whn_2);
    
    // onnx__GRU_206
    flag = read_weights(weights_path, "onnx__GRU_206.npy", &data_onnx__GRU_206, &size_onnx__GRU_206);
    if(flag != 0)
        return -1;
    
    data_biz_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_biz_2[i] = data_onnx__GRU_206[i];
    }
    data_biz_2_q = (BIZ_2_TYPE*) malloc(sizeof(BIZ_2_TYPE)*400);
    QUANTIZE(data_biz_2, data_biz_2_q, BIZ_2_S, BIZ_2_Z, 400)
    free(data_biz_2);

    data_bir_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bir_2[i] = data_onnx__GRU_206[400+i];
    }
    data_bir_2_q = (BIR_2_TYPE*) malloc(sizeof(BIR_2_TYPE)*400);
    QUANTIZE(data_bir_2, data_bir_2_q, BIR_2_S, BIR_2_Z, 400)
    free(data_bir_2);

    data_bin_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bin_2[i] = data_onnx__GRU_206[800+i];
    }
    data_bin_2_q = (BIN_2_TYPE*) malloc(sizeof(BIN_2_TYPE)*400);
    QUANTIZE(data_bin_2, data_bin_2_q, BIN_2_S, BIN_2_Z, 400)
    free(data_bin_2);

    data_bhz_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhz_2[i] = data_onnx__GRU_206[1200+i];
    }
    data_bhz_2_q = (BHZ_2_TYPE*) malloc(sizeof(BHZ_2_TYPE)*400);
    QUANTIZE(data_bhz_2, data_bhz_2_q, BHZ_2_S, BHZ_2_Z, 400)
    free(data_bhz_2);

    data_bhr_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhr_2[i] = data_onnx__GRU_206[1600+i];
    }
    data_bhr_2_q = (BHR_2_TYPE*) malloc(sizeof(BHR_2_TYPE)*400);
    QUANTIZE(data_bhr_2, data_bhr_2_q, BHR_2_S, BHR_2_Z, 400)
    free(data_bhr_2);

    data_bhn_2 = (float*) malloc(sizeof(float)*400);
    for(int i=0; i<400; i++) {
        data_bhn_2[i] = data_onnx__GRU_206[2000+i];
    }
    data_bhn_2_q = (BHN_2_TYPE*) malloc(sizeof(BHN_2_TYPE)*400);
    QUANTIZE(data_bhn_2, data_bhn_2_q, BHN_2_S, BHN_2_Z, 400)
    free(data_bhn_2);

    // onnx__MatMul_207
    flag = read_weights(weights_path, "onnx__MatMul_207.npy", &data_onnx__MatMul_207, &size_onnx__MatMul_207);
    if(flag != 0)
        return -1;
    TRANSPOSE(400, 600, data_onnx__MatMul_207, float)
    data_onnx__MatMul_207_q = (ONNX__MATMUL_207_TYPE*) malloc(sizeof(ONNX__MATMUL_207_TYPE) * size_onnx__MatMul_207);
    QUANTIZE(data_onnx__MatMul_207, data_onnx__MatMul_207_q, ONNX__MATMUL_207_S, ONNX__MATMUL_207_Z, size_onnx__MatMul_207)
    free(data_onnx__MatMul_207);
    
    // onnx__MatMul_208
    flag = read_weights(weights_path, "onnx__MatMul_208.npy", &data_onnx__MatMul_208, &size_onnx__MatMul_208);
    if(flag != 0)
        return -1;
    TRANSPOSE(600, 600, data_onnx__MatMul_208, float)
    data_onnx__MatMul_208_q = (ONNX__MATMUL_208_TYPE*) malloc(sizeof(ONNX__MATMUL_208_TYPE) * size_onnx__MatMul_208);
    QUANTIZE(data_onnx__MatMul_208, data_onnx__MatMul_208_q, ONNX__MATMUL_208_S, ONNX__MATMUL_208_Z, size_onnx__MatMul_208)
    free(data_onnx__MatMul_208);
    
    // onnx__MatMul_209
    flag = read_weights(weights_path, "onnx__MatMul_209.npy", &data_onnx__MatMul_209, &size_onnx__MatMul_209);
    if(flag != 0)
        return -1;
    TRANSPOSE(600, 257, data_onnx__MatMul_209, float)
    data_onnx__MatMul_209_q = (ONNX__MATMUL_209_TYPE*) malloc(sizeof(ONNX__MATMUL_209_TYPE) * size_onnx__MatMul_209);
    QUANTIZE(data_onnx__MatMul_209, data_onnx__MatMul_209_q, ONNX__MATMUL_209_S, ONNX__MATMUL_209_Z, size_onnx__MatMul_209)
    free(data_onnx__MatMul_209);
    
    
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

void run_nsnet2(float* x, float* h1, float* h2) {
    data_x_q = (X_TYPE*) malloc(sizeof(X_TYPE*) * size_x);
    QUANTIZE(x, data_x_q, X_S, X_Z, size_x)

    data_h1_q = (H1_TYPE*) malloc(sizeof(H1_TYPE*) * size_h1);
    QUANTIZE(h1, data_h1_q, H1_S, H1_Z, size_h1)

    data_h2_q = (H2_TYPE*) malloc(sizeof(H2_TYPE*) * size_h2);
    QUANTIZE(h2, data_h2_q, H2_S, H2_Z, size_h2)

    data_fc1MatMul_q = (FC1MATMUL_TYPE*) malloc(sizeof(FC1MATMUL_TYPE*) * size_fc1MatMul);
    QMATMUL(400, 257, data_onnx__MatMul_166_q, data_x_q, data_fc1MatMul_q, 
        ONNX__MATMUL_166_S,X_S,FC1MATMUL_S,
        ONNX__MATMUL_166_Z,X_Z,FC1MATMUL_Z, FC1MATMUL_TYPE)
    
    data_fc1Add_q = (FC1ADD_TYPE*) malloc(sizeof(FC1ADD_TYPE) * size_fc1Add);
    QADD(size_fc1Add, data_fc1MatMul_q, data_fc1_bias_q, data_fc1Add_q, FC1MATMUL_S, FC1_BIAS_S, FC1ADD_S, FC1MATMUL_Z, FC1_BIAS_Z, FC1ADD_Z)

    data_gru1_a__q = (GRU1_A__TYPE*) malloc(sizeof(GRU1_A__TYPE*) * size_gru1_a_);
    QMATMUL(400, 400, data_Wir_1_q, data_fc1Add_q, data_gru1_a__q, 
        WIR_1_S,FC1ADD_S,GRU1_A__S,
        WIR_1_Z,FC1ADD_Z,GRU1_A__Z,GRU1_A__TYPE)
    
    data_gru1_a_q = (GRU1_A_TYPE*) malloc(sizeof(GRU1_A_TYPE) * size_gru1_a);
    QADD(size_gru1_a, data_gru1_a__q, data_bir_1_q, data_gru1_a_q, 
        GRU1_A__S, BIR_1_S, GRU1_A_S, GRU1_A__Z, BIR_1_Z, GRU1_A_Z)
    
    data_gru1_b__q = (GRU1_B__TYPE*) malloc(sizeof(GRU1_B__TYPE*) * size_gru1_b_);
    QMATMUL(400, 400, data_Whr_1_q, data_h1_q, data_gru1_b__q, 
        WHR_1_S,H1_S,GRU1_B__S,
        WHR_1_Z,H1_Z,GRU1_B__Z,GRU1_B__TYPE)

    data_gru1_b_q = (GRU1_B_TYPE*) malloc(sizeof(GRU1_B_TYPE) * size_gru1_b);
    QADD(size_gru1_b, data_gru1_b__q, data_bhr_1_q, data_gru1_b_q, 
        GRU1_B__S, BHR_1_S, GRU1_B_S, GRU1_B__Z, BHR_1_Z, GRU1_B_Z)

    data_gru1_c__q = (GRU1_C__TYPE*) malloc(sizeof(GRU1_C__TYPE*) * size_gru1_c_);
    QMATMUL(400, 400, data_Wiz_1_q, data_fc1Add_q, data_gru1_c__q, 
        WIZ_1_S,FC1ADD_S,GRU1_C__S,
        WIZ_1_Z,FC1ADD_Z,GRU1_C__Z,GRU1_C__TYPE)
    
    data_gru1_c_q = (GRU1_C_TYPE*) malloc(sizeof(GRU1_C_TYPE) * size_gru1_c);
    QADD(size_gru1_c, data_gru1_c__q, data_biz_1_q, data_gru1_c_q, 
        GRU1_C__S, BIZ_1_S, GRU1_C_S, GRU1_C__Z, BIZ_1_Z, GRU1_C_Z)
    
    data_gru1_d__q = (GRU1_D__TYPE*) malloc(sizeof(GRU1_D__TYPE*) * size_gru1_d_);
    QMATMUL(400, 400, data_Whz_1_q, data_h1_q, data_gru1_d__q, 
        WHZ_1_S,H1_S,GRU1_D__S,
        WHZ_1_Z,H1_Z,GRU1_D__Z,GRU1_D__TYPE)
    
    data_gru1_d_q = (GRU1_D_TYPE*) malloc(sizeof(GRU1_D_TYPE) * size_gru1_d);
    QADD(size_gru1_d, data_gru1_d__q, data_bhz_1_q, data_gru1_d_q, 
        GRU1_D__S, BHZ_1_S, GRU1_D_S, GRU1_D__Z, BHZ_1_Z, GRU1_D_Z)
    
    data_gru1_e__q = (GRU1_E__TYPE*) malloc(sizeof(GRU1_E__TYPE*) * size_gru1_e_);
    QMATMUL(400, 400, data_Win_1_q, data_fc1Add_q, data_gru1_e__q, 
        WIN_1_S,FC1ADD_S,GRU1_E__S,
        WIN_1_Z,FC1ADD_Z,GRU1_E__Z,GRU1_E__TYPE)

    data_gru1_e_q = (GRU1_E_TYPE*) malloc(sizeof(GRU1_E_TYPE) * size_gru1_e);
    QADD(size_gru1_e, data_gru1_e__q, data_bin_1_q, data_gru1_e_q, 
        GRU1_E__S, BIN_1_S, GRU1_E_S, GRU1_E__Z, BIN_1_Z, GRU1_E_Z)

    data_gru1_f__q = (GRU1_F__TYPE*) malloc(sizeof(GRU1_F__TYPE*) * size_gru1_f_);
    QMATMUL(400, 400, data_Whn_1_q, data_h1_q, data_gru1_f__q, 
        WHN_1_S,H1_S,GRU1_F__S,
        WHN_1_Z,H1_Z,GRU1_F__Z,GRU1_F__TYPE)

    PRINT_TENSOR(data_gru1_f__q, 0, 10, "%d ", "\n")
    PRINT_TENSOR_SUM(data_gru1_f__q, 400, int, "%d\n")
}
