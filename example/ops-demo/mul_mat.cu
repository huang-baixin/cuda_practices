// demo.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(float* d_out, const float* d_in)
{
    int idx = threadIdx.x;
    float a = d_in[idx];
    float b = d_in[idx] * 2.0f;
    d_out[idx] = a + b;
}

int main()
{
    const int ARRAY_SIZE = 1024;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array in host memory
    float h_in[ARRAY_SIZE], h_out[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // declare GPU memory pointers
    float * d_in, * d_out;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    simpleKernel<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy the output array back from the GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the result
    std::cout << "Output: ";
    for(int i = 0; i < ARRAY_SIZE; i++){
        if(i < 5) std::cout << h_out[i] << " ";
    }
    std::cout << "..." << std::endl;

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}