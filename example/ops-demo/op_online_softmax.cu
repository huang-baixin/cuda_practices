// demo.cu
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
#define BLOCK_SIZE 128


struct __align__(8) M_NF {
    float max;
    float norm_factor;
};


struct M_NF_op {
    __device__ __forceinline__ M_NF operator()(M_NF &a, M_NF &b) {
        M_NF ret;
        ret.max = max(a.max, b.max);
        ret.norm_factor = a.norm_factor * __expf(a.max - ret.max) + b.max * __expf(b.max - ret.norm_factor);
        return ret;
    }
};


__global__ void online_softmax_compute(const float*  __restrict__ dst, float* __restrict__ src, const int n_col) {

    // find mx value

    // calculate normalization factor


    // calculate softmax  


}





__global__ void three_pass_softmax_compute(float* dst, float* src, const size_t n_row, const size_t n_col, cudaStream_t stream) {


    float cur;
    float max_elem = FLT_MIN;
    float norm_factor = 1e-10f;

    // find mx value
#pragma unroll
    for(int i = threadIdx.x; i < n_col; i+= blockDim.x) {
        max_elem = max(src[blockIdx.x * n_col + i], max_elem);
    }
    __syncthreads();

    max_elem = blockAllReduceMax<float>(vmax);

    // calculate normalization factor
#pragma unroll
    for(int i = threadIdx.x; i < n_col; i+= blockDim.x) {
        norm_factor += __expf(src(blockIdx.x * n_col + i) - max_elem);
    }
    __syncthreads();

    norm_factor = blockAllReduceSum<float>(norm_factor);

    // calculate softmax  
#pragma unroll
    for(int i = threadIdx.x; i < n_col; i+= blockDim.x) {

        cur = __expf(src[blockIdx.x * n_col + i] - max_elem) / norm_factor;
        dst[blockIdx.x * n_col + i] = cur;
    }

}



void __global__ launch_softmax(float* dst, float* src, const size_t n_row, const size_t n_col, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(n_row)
    online_softmax_compute<<<grid, block, 0, stream>>>(dst, src, n_col);
}



int main()
{

    size_t n_col = 128;
    size_t n_row = 128;
    size_t alloc_num = n_col * n_row;
    size_t alloc_size = alloc_num * sizeof(float);

    float* host_in = (float*)calloc(alloc_num);
    float* host_out = (float*)calloc(alloc_num);
    
    float * d_in, * d_out;

    cudaMalloc((void**) &d_in, alloc_size);
    cudaMalloc((void**) &d_out, alloc_size);

    cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;

    launch_online_softmax(d_in, d_out, n_row, n_col, stream);

    cudaMemcpy(host_out, d_out, alloc_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(host_data);

    return 0;
}