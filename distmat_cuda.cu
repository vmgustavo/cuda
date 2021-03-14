#include <R.h>
#include <math.h>
#include <cuda.h>
#include "stdio.h"

#define BLOCK_WIDTH 16


extern "C"

__host__ void distmat_cuda(double *arr, int *feats, int *samples, double *res);
__global__ void distmat_kernel(double* arr, int feats, int samples, double* res);


int main (void) { return 0; }


__host__ void distmat_cuda(double *arr, int *feats, int *samples, double *res) {
    double* device_arr;
    double* device_res;

    int f = *feats;
    int s = *samples;
    int size;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // COPY DATA TO DEVICE
    size = f * s * sizeof(double);
    cudaStatus = cudaMalloc((void**)&device_arr, size);
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaMalloc failed!");
    }
    cudaStatus = cudaMemcpy(device_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaMemcpy failed!");
    }

    // ALLOCATE RESULT DISTANCE MATRIX
    size = s * s * sizeof(double);
    cudaStatus = cudaMalloc((void**)&device_res, size);
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaMalloc failed!");
    }

    // SETUP CUDA KERNEL
    dim3 threadsPerBlock(8, 8, 1);
    dim3 numBlocks(
        ceil(double(s) / threadsPerBlock.x),
        ceil(double(s) / threadsPerBlock.y),
        1
    );

    Rprintf("threadsPerBlock | %d : %d : %d\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    Rprintf("numBlocks       | %d : %d : %d\n", numBlocks.x, numBlocks.y, numBlocks.z);

    // int numBlocks = ceil(double(s) / double(BLOCK_WIDTH));
    // dim3 dimGrid(numBlocks, numBlocks, 1);
    // dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // printf("numBlock\t%d\n", numBlocks);
    // printf("dimGrid\t\t%d : %d : %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    // printf("dimBlock\t%d : %d : %d\n", dimBlock.x, dimBlock.y, dimBlock.z);

    // STARTS KERNEL
    // distmat_kernel << < dimGrid, dimBlock >> > (device_arr, f, s, device_res);
    distmat_kernel << < threadsPerBlock, numBlocks >> > (device_arr, f, s, device_res);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        Rprintf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // WAITS FOR THE KERNEL TO FINISH
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // TRANSFER DATA FROM DEVICE TO HOST
    size = s * s * sizeof(double);
    cudaStatus = cudaMemcpy(res, device_res, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaMemcpy failed!");
    }

    // FREE DEVICE
    cudaFree(device_arr);
    cudaFree(device_res);

    Rprintf("%s\n", cudaGetErrorString(cudaStatus));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        Rprintf("cudaDeviceReset failed!");
    }
    
}


__global__ void distmat_kernel(double* arr, int feats, int samples, double* res) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    uint k;
    double aux = 0;
    for (k = 0; k < feats; k++) {
        // loop through features
        double diff = arr[i + samples * k] - arr[j + samples * k];
        aux += pow(diff, 2);
    }

    res[i + samples * j] = sqrt(aux);
    // res[i + samples * j] = 1.0;
}
