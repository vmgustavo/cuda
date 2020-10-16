#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_WIDTH 32
#define TILE_WIDTH 32

__host__ void h_mat_mul(float* A, float* B, float* C, int n);
__host__ void d_mat_mul(float* A, float* B, float* C, int n);
__host__ void mat_mul_kernel(float* A, float* B, float* C, int n);

void printVector(float *x, int n) {
    printf("| ");
    for (int i = 0; i < n; i++)
        printf("%.1f | ", x[i]);
    printf("\n");
}

int main()
{
	printf("start program\n");
    int n = 1000;
    printf("multiplying %dX%d matrices\n", n, n);

	float *a = new float[n * n];
	float *b = new float[n * n];
	float *c = new float[n * n];

	clock_t st;
	clock_t en;

	int i = 0;
	for (i = 0; i < n * n; i++)
	{
		a[i] = b[i] = 1;
		c[i] = 0;
	}

	int n_to_print = 4;
    printVector(a, n_to_print);
    printVector(b, n_to_print);
    printVector(c, n_to_print);

    // ========================================================================
	// multiply matrices with host function
    printf("\nhost function\n");
    st = clock();
    h_mat_mul(a, b, c, n);
    en = clock();

    printf("Elapsed: %.03f seconds\n", (double)(en - st) / CLOCKS_PER_SEC);
    printVector(c, n_to_print);

	for (int i = 0; i < n; i++) {
		a[i] = b[i] = 1;
		c[i] = 0;
    }

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

__host__ void h_mat_mul(float* a, float* b, float* c, int n) {
    // iterate over the rows of the output
    for (int i = 0; i < n; i++) {
        // iterate over the columns of the output
		for (int j = 0; j < n; j++) {
            // iterate over the values of the input of a given row an col
		    for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[j * n + k];
		    }
		}
	}
}

__host__ void d_mat_mul(float* a, float* b, float* c, int n) {
    int size = n * n * sizeof(float);
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
    dim3 dimGrid(num_blocks, BLOCK_WIDTH, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

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

__global__ void matrix_mul_kernel(float* A, float* B, float* C, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < n) && (col < n))
	{
		float cValue = 0;
		for (int k = 0; k < n; ++k)
			cValue += A[row*n + k] * B[k*n + col];
		C[row*n + col] = cValue;
	}
}