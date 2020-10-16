#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_WIDTH 32
#define TILE_WIDTH 32

__host__ void h_vec_add(float* a, float* b, float* c, int n);
__host__ void d_vec_add(float* a, float* b, float* c, int n);
__global__ void vec_add_kernel(float* a, float* b, float* c, int n);

void printVector(float *x, int n) {
    printf("| ");
    for (int i = 0; i < n; i++)
        printf("%.1f | ", x[i]);
    printf("\n");
}

int main() {
    printf("start program\n");
    int n = 100000000;
    printf("adding %d elements\n", n);

	float *a = new float[n];
	float *b = new float[n];
    float *c = new float[n];

    clock_t st;
    clock_t en;

	for (int i = 0; i < n; i++) {
		a[i] = b[i] = 1;
		c[i] = 0;
    }

    int n_to_print = 4;
    printVector(a, n_to_print);
    printVector(b, n_to_print);
    printVector(c, n_to_print);

    // ========================================================================
    // add vectors with host function
    printf("\nhost function\n");
    st = clock();
    h_vec_add(a, b, c, n);
    en = clock();

    printf("Elapsed: %.03f seconds\n", (double)(en - st) / CLOCKS_PER_SEC);
    printVector(c, n_to_print);

	for (int i = 0; i < n; i++) {
		a[i] = b[i] = 1;
		c[i] = 0;
    }

    // ========================================================================
    // add vectors with device function
    printf("\ndevice function\n");
    st = clock();
    d_vec_add(a, b, c, n);
    en = clock();

    printf("Elapsed: %.03f seconds\n", (double)(en - st) / CLOCKS_PER_SEC);
    printVector(c, n_to_print);
}

__host__ void h_vec_add(float* a, float* b, float* c, int n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__host__ void d_vec_add(float* a, float* b, float* c, int n) {
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
    
    // printf("numBlock\t%d\n", num_blocks);
	// printf("dimGrid\t\t%d : %d : %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    // printf("dimBlock\t%d : %d : %d\n", dimBlock.x, dimBlock.y, dimBlock.z);

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
