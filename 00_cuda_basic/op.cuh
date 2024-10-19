#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

template<typename T>
void reduce_recursive_host(T* data, size_t size) {
    if (size == 0)  { return data[0]; }
    const int strie = size / 2;
    for (int idx = 0; idx < stride; ++idx) {
        data[idx] = data[idx + stride];
    }
    return reduce_recursive_host(data, stride); // call recursive
}


// 就是 cpu reduce 的简单实现
__global__ void reduce_neighbored(float* src0, float* dst) {
    // we do the reduce in only one block
    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float* bptr = blockIdx.x * blockDim;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (stride * 2) == 0) {
            bptr[tid] += bptr[tid + stride];
        }
    }
    __syncthreads();
    if (tid == 0) dst[blockIdx.x] = bptr[0];
}

__global__ void reduce_interleaved(float* src0, float* dst) {
    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float* bptr = blockIdx.x * blockDim.x;
    float* bptr = blockIdx.x * blockDim;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 2) {
        bptr[tid] = bptr[tid + stride];
    }
    __syncthreads();
    if (tid == 0) dst[blockIdx.x] = bptr[0];
}

//  
__global__ void reduce_neighboredless(float* src0, float* dst) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 
    float* bptr = blockIdx.x * blockDim;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            bptr[index] += bptr[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) dst[blockIdx.x] = bptr[0];
}



