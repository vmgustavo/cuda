#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <r.h>

__host__ void vecadd_cuda(float* a, float* b, float* c, int n) {
    int size = n * sizeof(float);
    float* aD;
    float* bD;
    float* cD;

    // allocate memory for vector and copy to device
    cudaMalloc((void**)&aD, size);
    cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bD, size);
    cudaMemcpy(bD, b, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cD, size);
    cudaMemcpy(cD, c, size, cudaMemcpyHostToDevice);

    // setup cuda kernel
    int num_blocks = ceil(n / BLOCK_WIDTH);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, 1, 1);

    // start cuda kernel
    vec_add_kernel << < dimGrid, dimBlock >> > (aD, bD, cD, n);

	// waits for the kernel to finish and returns errors
	cudaDeviceSynchronize();

	// copy c from device
	cudaMemcpy(c, cD, size, cudaMemcpyDeviceToHost);

	// free device
	cudaFree(aD);
	cudaFree(bD);
	cudaFree(cD);
}

__global__ void vec_add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
