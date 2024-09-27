#include <stdio.h>
#include "q_nsnet2.h"



int main() {
    int flag;
    char* weights_path = "/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/";
    flag = setup_nsnet2(weights_path);
    if(flag != 0) {
        printf("Error in nsnet2 initialization\n");
        return -1;
    }
    free_nsnet2();
    return 0;
}