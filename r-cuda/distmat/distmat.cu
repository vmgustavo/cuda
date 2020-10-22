#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_WIDTH 32
#define TILE_WIDTH 32

__host__ void h_distmat(float* feat, float* dist, int n);
__host__ void d_distmat(float* feat, float* dist, int n);
__global__ void kernel_distmat(float* feat, float* dist, int n);

void printVector(float *x, int n) {
    printf("| ");
    for (int i = 0; i < n; i++)
        printf("%.1f | ", x[i]);
    printf("\n");
}

void setup_values(float *features, float *dist, int n, int n_dist) {
    for (int i = 0; i < n; i++) features[i] = i;
    for (int i = 0; i < n_dist; i++) dist[i] = 0;
}

int main() {
    printf("start program\n");
    int n = 10000;
    float *features = new float[n];

    int n_dist = 0;
    for (int i = n; i > 0; i--) n_dist += i;
    float *dist = new float[n_dist];
    printf("n: %d | n_dist: %d\n", n, n_dist);

    clock_t st;
    clock_t en;

    setup_values(features, dist, n, n_dist);

    int n_to_print = 4;
    printVector(features, n_to_print);
    printVector(dist, n_to_print);

    // ========================================================================
    // HOST FUNCTION
    printf("\nhost function\n");
    st = clock();
    h_distmat(features, dist, n);
    en = clock();

    printf("Elapsed: %.03f seconds\n", (double)(en - st) / CLOCKS_PER_SEC);
    printVector(dist, n_to_print);

    // reset values
    setup_values(features, dist, n, n_dist);

    // ========================================================================
    // DEVICE FUNCTION
    printf("\ndevice function\n");
    st = clock();
    d_distmat(features, dist, n);
    en = clock();

    printf("Elapsed: %.03f seconds\n", (double)(en - st) / CLOCKS_PER_SEC);
    printVector(dist, n_to_print);
}

__host__ void h_distmat(float* feat, float* dist, int n) {
    /*
     * feat: features describing the sample / currently 1D only
     * dist: condensed distance matrix
     * n: number of samples in the feat array
    */
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            // printf("i: %d | j: %d | dcoord: %d | feat[j]: %.0f | feat[i]: %.0f\n", i, j, count, feat[j], feat[i]);
            dist[count] = feat[j] - feat[i];
            count ++;
        }
    }
}

__host__ void d_distmat(float* feat, float* dist, int n) {
   int size = n * sizeof(float);
   float* featD;
   float* distD;

   // allocate memory for vector and copy to device
   cudaMalloc((void**)&featD, size);
   cudaMemcpy(featD, feat, size, cudaMemcpyHostToDevice);

   cudaMalloc((void**)&distD, size);
   cudaMemcpy(distD, dist, size, cudaMemcpyHostToDevice);

   // setup cuda kernel
   int num_blocks = ceil(n / BLOCK_WIDTH);
   dim3 dimGrid(num_blocks, num_blocks, 1);
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

   // printf("numBlock\t%d\n", num_blocks);
   // printf("dimGrid\t\t%d : %d : %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
   // printf("dimBlock\t%d : %d : %d\n", dimBlock.x, dimBlock.y, dimBlock.z);

   // start cuda kernel
   kernel_distmat << < dimGrid, dimBlock >> > (featD, distD, n);

   // waits for the kernel to finish and returns errors
   cudaDeviceSynchronize();

   // copy c from device
   cudaMemcpy(dist, distD, size, cudaMemcpyDeviceToHost);

   // free device
   cudaFree(featD);
   cudaFree(distD);
}

__global__ void kernel_distmat(float* feat, float* dist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n) && (j < n) && (j >= i)){
        dist[i * n + j] = feat[j] - feat[i];
    }
}
