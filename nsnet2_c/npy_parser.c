#include "npy_parser.h"

int read_npy_file(FILE *file, float** data, int* size) {
    char magic[6];
    fread(magic, sizeof(char), 6, file);
    magic[6] = '\0';
    
    if (strcmp(magic, "\x93NUMPY") != 0) {
        printf("Error: this is not a numpy file\n");
        return -1;
    }
    
    unsigned char version[2];
    fread(version, sizeof(unsigned char), 2, file);
    //printf("Version .npy: %d.%d\n", version[0], version[1]);
    
    unsigned short header_len;
    fread(&header_len, sizeof(unsigned short), 1, file);
    
    char *header = (char *)malloc(header_len + 1);
    fread(header, sizeof(char), header_len, file);
    header[header_len] = '\0';
    
    //printf("Header: %s\n", header);
    
    char *descr = strstr(header, "'descr': '");
    char dtype[10];
    if (descr != NULL) {
        sscanf(descr, "'descr': '%[^']'", dtype);
        //printf("Dtype: %s\n", dtype);
    }
    
    char *shape_str = strstr(header, "'shape': (");
    int shape[10];
    int ndim = 0;
    if (shape_str != NULL) {
        char *ptr = shape_str + strlen("'shape': (");
        char *ptr_ = ptr;
        int len_str = strlen(ptr);
        int start = 0, end;
        for(int i=0; i<len_str; i++) {
            if(*ptr == ',' || *ptr == ')') {
                end = i;
                char* substr = (char*) malloc(sizeof(char) * (end - start));
                for(int j=0; j<(end-start); j++)
                    substr[j] = ptr_[start+j];
                int dim = atoi(substr);
                shape[ndim++] = dim;
                free(substr);
                if(*ptr == ')' || (*ptr == ',' && *(ptr+1) == ')'))
                    break;
                else
                    start = i+1;
            }
            ptr++;
        }
    }
    
    free(header);

    //printf(descr);
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    *size = total_size;
    *data = (float *)malloc(total_size * sizeof(float));
    fread(*data, sizeof(float), total_size, file);
    /*for(int i=0; i<total_size; i++) {
        printf("%d ", i);
        printf("%f ", (*data)[i]);
    }*/

    /*printf("Data:\n");
    for (int i = 0; i < total_size; i++) {
        printf("%f ", data[i]);
        if ((i + 1) % shape[ndim - 1] == 0) {
            printf("\n");
        }
    }*/

    //free(data);
    return 0;
}