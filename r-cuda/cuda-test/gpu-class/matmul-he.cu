#include <stdio.h>
#include "cuda_runtime.h"
#define BLOCK_SIZE 3
#define TILE_WIDTH 3
#define WIDTH 4


__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    if ((Row < Width) && (Col < Width)){
        float Pvalue = 0;
        // Loop over the d_M and d_N tiles required to compute d_P element
        for (int m = 0; m < Width / TILE_WIDTH; ++m) {
            // Coolaborative loading of d_M and d_N tiles into shared memory
            Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
            Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
            __syncthreads();
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
        d_P[Row*Width + Col] = Pvalue;
    }
}

__global__ void MatrixMulKernelS(float* d_M, float* d_N, float* d_P, int Width) {
    // Calculate the row index of the d_P element and d_M
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    // Calculate the column index of d_P and d_N
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k) {
            Pvalue += d_M[Row*Width + k] * d_N[k*Width + Col];
        }
        d_P[Row*Width + Col] = Pvalue;
    }
}
//FILLRANDOM
//Given the address of a matrix (m x n), the function generates random numbers (between 0 and 99) for its elements
void fillRandom(float *M, int m, int n){
    int i;

    for (i = 0; i < (m*n); i++){
        M[i] = (float)(rand() % 10);
        //printf("%d ", M[i]);
    }
    return;
}

int main(){


    //Matrices A and B: WIDTH x WIDTH elements
    srand(time(NULL));
    float *A = (float *)malloc(WIDTH * WIDTH * sizeof(float));
    fillRandom(A, WIDTH, WIDTH);
    float *B = (float *)malloc(WIDTH * WIDTH * sizeof(float));
    fillRandom(B, WIDTH, WIDTH);
    float *C;
    C = (float *)malloc(WIDTH* WIDTH * sizeof(float));

    int i = 0;

    for (i = 0; i < WIDTH*WIDTH; i++){
        printf("%.2f ", *(A + i));
        if ((i%WIDTH) == (WIDTH - 1))
            putchar('\n');
    }
    putchar('\n');

    for (i = 0; i < WIDTH*WIDTH; i++){
        printf("%.2f ", *(B + i));
        if ((i%WIDTH) == (WIDTH - 1))
            putchar('\n');
    }

    putchar('\n');

    //Allocating memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&d_B, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&d_C, WIDTH * WIDTH * sizeof(float));
    //Copying data from host to device
    cudaMemcpy(d_A, A, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(WIDTH / BLOCK_SIZE), ceil(WIDTH / BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    MatrixMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C, WIDTH);
    cudaMemcpy(C, d_C, WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost);


    printf("\nRESULTS: \n");
    for (i = 0; i < WIDTH*WIDTH; i++){
        printf("%.2f ", *(C + i));
        if ((i%WIDTH) == (WIDTH - 1))
            putchar('\n');
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
