#include <R.h>
#include <math.h>
#include <cuda.h>

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

    // COPY DATA TO DEVICE
    size = f * s * sizeof(double);
    cudaMalloc((void**)&device_arr, size);
    cudaMemcpy(device_arr, arr, size, cudaMemcpyHostToDevice);

    // ALLOCATE RESULT DISTANCE MATRIX
    size = s * s * sizeof(double);
    cudaMalloc((void**)&device_res, size);
    cudaMemcpy(device_res, res, size, cudaMemcpyHostToDevice);

    // SETUP CUDA KERNEL
    int num_blocks = samples;
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, 1, 1);

    // STARTS KERNEL
    distmat_kernel << < dimGrid, dimBlock >> > (device_arr, f, s, device_res);

    // WAITS FOR THE KERNEL TO FINISH
    cudaDeviceSynchronize();

    // TRANSFER DATA FROM DEVICE TO HOST
    size = s * s * sizeof(double);
    cudaMemcpy(res, device_res, size, cudaMemcpyDeviceToHost);

    // FREE DEVICE
    cudaFree(device_arr);
    cudaFree(device_res);

    // res[0] = 1;
}


__global__ void distmat_kernel(double* arr, int feats, int samples, double* res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    res[i + samples * j] = i;
}
