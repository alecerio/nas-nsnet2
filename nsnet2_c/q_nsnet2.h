#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int setup_nsnet2(const char* weights_path);
void free_nsnet2();

static int read_weights(const char* weights_path, const char* weights_name, float** data, int* size);