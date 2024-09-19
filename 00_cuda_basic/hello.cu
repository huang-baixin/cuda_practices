#include <stdio.h>
#include "../common.h"

__global__ void hello_world() {
    printf("hello world\n");
}



int main() {
    hello_world<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}
