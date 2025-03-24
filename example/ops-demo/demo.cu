#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <time.h>


#define OFFSET(cur_row, cur_col, row_size) ((cur_row * row_size) + (cur_col))


__global__ void print_dims() {
    printf("%3d, %3d, %3d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("%3d, %3d, %3d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("%3d, %3d, %3d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("%3d, %3d, %3d\n", gridDim.x, gridDim.y, gridDim.z);
}

__global__ void test_shfl_xor(int A[], int B[])
{
    int tid = threadIdx.x;
    int best = B[tid];
    int mask = 16;
    best = __shfl_xor_sync(0xffffffff, best, mask, 8);
    A[tid] = best;
}

__global__ void bcast2(float* a, float* b) {
    int laneId = threadIdx.x;
    float value;
    for (int mask = 16; mask > 0; mask >>= 1) {
        a[laneId] += __shfl_xor_sync(0xffffffff, a[laneId], mask, 32);
        b[laneId] += __shfl_xor_sync(0xffffffff, b[laneId], mask, 32);
    }
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}


// CUDA内核函数：对两个数组进行加法运算
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // what should we consider ehile wo are coding a kernel?
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 - 255
    C[idx] = A[idx] + B[idx];
}

__global__ void mat_add(const float* A, const float* B, float* C, int cols, int rows) {
    int cur_col = blockIdx.x * blockDim.x + threadIdx.x;
    int cur_row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = cur_row * cols + cur_col;
    C[idx] = A[idx] + B[idx];
}


__global__ void mul_mat_simple(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int M, const int N, const int K) {
    // [M, K] x [K, N]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float psum = 0.0f;
    
    for (int k = 0; k < K; ++k) {
        // A[row, k], B[k, col]
        psum = A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)];
    }
    C[OFFSET(row, col, N)] = psum;
}


__global__ void softmax_simple_f32_2(const float* src0, float* dst, int size, bool inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_elem = -INFINITY;
    for (int i = 0; i < size; ++i) {
        max_elem = fmaxf(src0[i], max_elem);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += expf(src0[i] - max_elem);
    }

    dst[idx] = expf(src0[idx] - max_elem) / sum;
}





int mul_mat_demo() {  
    int N = 4096;
    size_t size = N * sizeof(float);  

    float* h_A = (float*)malloc(size);  
    float* h_B = (float*)malloc(size);  
    float* h_C = (float*)malloc(size);  

    for (int i = 0; i < N; ++i) {  
        h_A[i] = i;  
        h_B[i] = i * 2.0f; // 例如，让B为A的两倍  
    }  

    float *d_A, *d_B, *d_C;  
    cudaMalloc(&d_A, size);  
    cudaMalloc(&d_B, size);  
    cudaMalloc(&d_C, size);  

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  

    {
        int threadsPerBlock = 256; // 每个块中的线程数  
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 总块数  
    }

    {
        int cols  = 1024;
        int rows= 4;
        dim3 threadsPerBlock(64, 4, 1);
        dim3 blocksPerGrid( (cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
        mat_add<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_B, d_C, cols, rows);

    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  

    for (int i = 0; i < 4096; i+=256) {  
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl; // 输出前10个结果  
    }  

    cudaFree(d_A);  
    cudaFree(d_B);  
    cudaFree(d_C);  
    free(h_A);  
    free(h_B);  
    free(h_C);  

    return 0;  
} 

int vec_add_demo() {  
    int N = 4096;
    size_t size = N * sizeof(float);  

    float* h_A = (float*)malloc(size);  
    float* h_B = (float*)malloc(size);  
    float* h_C = (float*)malloc(size);  

    for (int i = 0; i < N; ++i) {  
        h_A[i] = i;  
        h_B[i] = i * 2.0f; // 例如，让B为A的两倍  
    }  

    // 在设备上分配内存  
    float *d_A, *d_B, *d_C;  
    cudaMalloc(&d_A, size);  
    cudaMalloc(&d_B, size);  
    cudaMalloc(&d_C, size);  

    // 将数据从主机拷贝到设备  
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  

    {
        // 256 threads per block (block_size)
        int threadsPerBlock = 256; // 每个块中的线程数  
        // 16 block per grid (grid_size)
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 总块数  
        // vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);  
    }

    {
        // [1024, 4] + [1024, 4]
        int cols  = 1024;
        int rows= 4;
        dim3 threadsPerBlock(64, 4, 1);
        dim3 blocksPerGrid( (cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
        mat_add<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_B, d_C, cols, rows);

    }
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  
    for (int i = 0; i < 4096; i+=256) {  
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl; // 输出前10个结果  
    }  
    // 释放设备和主机内存  
    cudaFree(d_A);  
    cudaFree(d_B);  
    cudaFree(d_C);  
    free(h_A);  
    free(h_B);  
    free(h_C);  

    return 0;  
} 




// TODO: kernal elapsed 
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

void test_max_mem_size() {
    size_t size = 1;
    size_t maxSize = 0;
    void *d_ptr = nullptr;
    while (true) {
        checkCudaErrors(cudaMalloc(&d_ptr, size));
        maxSize = size;
        printf("Allocated %.3f GB\n", size / (1024.f * 1024 * 1024));
        size += 512 * 1024 * 1024;
        checkCudaErrors(cudaFree(d_ptr));
    }
    printf("Maximum global memory allocated: %zu bytes\n", maxSize);
}



int main() {

    {
        dim3 grid(4096, 1, 1);
        dim3 block(1, 1, 1);
        vec_add_demo();
    }
    return 0;
}
