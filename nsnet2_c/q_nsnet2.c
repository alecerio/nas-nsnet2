#include "q_nsnet2.h"

// fc1_bias
static float* data_fc1_bias;
static int size_fc1_bias;

// fc2_bias
static float* data_fc2_bias;
static int size_fc2_bias;

int setup_nsnet2(const char* weights_path) {
    int flag;

    // fc1_bias
    const char* weights_name = "fc1_bias.npy";
    flag = read_weights(weights_path, "fc1_bias.npy", &data_fc1_bias, &size_fc1_bias);
    if(flag != 0)
        return -1;
    
    
    /*printf("size: %d\n", size_fc1_bias);
    for(int i=0; i<size_fc1_bias; i++) {
        printf("%f ", data_fc1_bias[i]);
    }*/

    return 0;
}

void free_nsnet2() {
    free(data_fc1_bias);
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
