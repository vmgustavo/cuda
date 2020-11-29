
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_WIDTH 16
#define TILE_WIDTH 16

//matrix multiplication kernel
__global__
void matrixMulKernel(float* A, float* B, float* C, int n);

//matrix multiplication handling
__host__
void matrixMul(float* a, float* b, float* c, int n);

int main()
{
    //memory allocation
    int n = 320;

    float *a = new float[n*n];
    float *b = new float[n*n];
    float *c = new float[n*n];

    //setting matrix
    int i = 0;
    for (i = 0; i < n*n; i++)
    {
        a[i] = b[i] = 1;
        c[i] = 0;
    }

    matrixMul(a, b, c, n);

    //int offset = n*n / (10);
    //printf("10 elements of a\n\t");
    //for (int i = 0; i < n*n; i += offset)
    //	printf("%.1f | ", a[i]);
    //printf("\n");
    //printf("10 elements of b\n\t");
    //for (int i = 0; i < n*n; i += offset)
    //	printf("%.1f | ", b[i]);
    //printf("\n");
    //printf("10 elements of c\n\t");
    //for (int i = 0; i < n*n; i += offset)
    //	printf("%.1f | ", c[i]);
    //printf("\n");

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

//matrix multiplication handling
__host__
void matrixMul(float* a, float* b, float* c, int n)
{
    int size = n*n*sizeof(float);
    float* aD;
    float* bD;
    float* cD;

    //allocate memory for matrix a
    cudaMalloc((void**)&aD, size);

    //copy matrix a to device
    cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice);

    //allocate memory for matrix b
    cudaMalloc((void**)&bD, size);

    //copy matrix b to device
    cudaMemcpy(bD, b, size, cudaMemcpyHostToDevice);

    //allocate memory for matrix c
    cudaMalloc((void**)&cD, size);

    //kernel launch
    int numBlocks = ceil(n / BLOCK_WIDTH);
    dim3 dimGrid(numBlocks, numBlocks, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    //printf("numBlockk\t%d\n", numBlocks);
    //printf("dimGrid\t\t%d : %d : %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //printf("dimBlock\t%d : %d : %d\n", dimBlock.x, dimBlock.y, dimBlock.z);

    matrixMulKernel << < dimGrid, dimBlock >> > (aD, bD, cD, n);

    //waits for the kernel to finish and returns errors
    cudaDeviceSynchronize();

    //copy c from device
    cudaMemcpy(c, cD, size, cudaMemcpyDeviceToHost);

    //free device
    cudaFree(aD);
    cudaFree(bD);
    cudaFree(cD);
}

//matrix multiplication kernel
__global__
void matrixMulKernel(float* A, float* B, float* C, int n)
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