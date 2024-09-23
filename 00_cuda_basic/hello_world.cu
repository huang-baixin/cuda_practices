#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "../common.h"

__global__ void hello_world() {
    printf("hello world\n");
}

__global__ void print_dims() {
    printf("%3d, %3d, %3d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("%3d, %3d, %3d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("%3d, %3d, %3d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("%3d, %3d, %3d\n", gridDim.x, gridDim.y, gridDim.z);
}


// add [4090, 4096]
__global__ void vector_add_f32(float* src1, float* src2, float* dst) {

    
    

}




void host_mem_alloc_and_init() {

}

// type
void mem_host_to_device(void* dst, void* src, size_t size) {


}

// CUDA内核函数：对两个数组进行加法运算
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // what should we consider ehile wo are coding a kernel?
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 - 255
    C[idx] = A[idx] + B[idx];
    
}


int vec_add_demo() {  
    int N = 4096;
    size_t size = N * sizeof(float);  

    // 在主机上分配内存  
    float* h_A = (float*)malloc(size);  
    float* h_B = (float*)malloc(size);  
    float* h_C = (float*)malloc(size);  

    // 初始化输入数组  
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

    // 256 threads per block (block_size)
    int threadsPerBlock = 256; // 每个块中的线程数  
    // 16 block per grid (grid_size)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 总块数  

    // 启动CUDA内核  
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);  

    // 从设备拷贝结果回主机  
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  

    // 打印结果的前10个元素  
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



int main() {
    // hello_world<<<1, 10>>>();
    // CHECK(cudaDeviceReset());
    {
        dim3 grid(3, 1, 1);
        dim3 block(2, 1, 1);
        // print_dims<<<grid, block>>>();
        CHECK(cudaDeviceReset());
    }

    {
        dim3 grid(4096, 1, 1);
        dim3 block(1, 1, 1);
        
        // vector_add_f32<<<grid, block>>>();
        vec_add_demo();
        
    }
    
    return 0;
}
