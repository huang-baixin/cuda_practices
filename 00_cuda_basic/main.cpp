#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <time.h>

#include "../common.h"
#include "op.cuh"

#define OFFSET(cur_row, cur_col, row_size) ((cur_row * row_size) + (cur_col))

int main() {
    {
        dim3 grid(3, 1, 1);
        dim3 block(2, 1, 1);
        CHECK(cudaDeviceReset());
    }

    {
        dim3 grid(4096, 1, 1);
        dim3 block(1, 1, 1);
        vec_add_demo();
    }
    
    return 0;
}
