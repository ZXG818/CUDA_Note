#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _INIT_MATRIX_(ret_type, func_name) ret_type func_name
#define _PRINT_MATRIX_(ret_type, func_name) ret_type func_name

// initialize the matrix 
_INIT_MATRIX_(void, InitMatrix)(float *A, int nx, int ny)
{
	int i, j;
	float cnt = 0;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			A[i*nx + j] = ++cnt;
		}
	}
}

// print the matrix 
_PRINT_MATRIX_(void, PrintMatrix)(float *A, int nx, int ny)
{
	int i, j;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			printf("%.2f  ", A[i*nx + j]);
		}
		printf("\n");
	}
}

// add matrix on CPU
void SumMatrixOnCPU(float *A, float *B, float *C, int nx, int ny)
{
	int i, j;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			C[i*nx + j] = A[i*nx + j] + B[i*nx + j];
		}
	}
}

// add the matrix on GPU
__global__ void SumMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = iy * nx + ix;
	if (ix < nx && iy < ny)
	{
		C[idx] = A[idx] + B[idx];
	}
}

int main(int argc, char *argv[])
{
	int N = 1 << 10;
	int nx = 1 << 5;
	int ny = 1 << 5;

	float *h_A = NULL;
	float *h_B = NULL;
	float *h_C = NULL;
	float *gpu_result = NULL;

	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;

	// allocate the memory on CPU
	h_A = (float *)malloc(sizeof(float)*N);
	h_B = (float *)malloc(sizeof(float)*N);
	h_C = (float *)malloc(sizeof(float)*N);
	gpu_result = (float *)malloc(sizeof(float)*N);
	memset(h_A, 0, sizeof(float)*N);
	memset(h_B, 0, sizeof(float)*N);
	memset(h_C, 0, sizeof(float)*N);
	memset(gpu_result, 0, sizeof(float)*N);

	// allocate the memory on GPU
	cudaMalloc((float **)&d_A, sizeof(float)*N);
	cudaMalloc((float **)&d_B, sizeof(float)*N);
	cudaMalloc((float **)&d_C, sizeof(float)*N);
	//cudaMemset(d_A, 0, N);
	//cudaMemset(d_B, 0, N);
	//cudaMemset(d_C, 0, N);

	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


	// make the initialization
	InitMatrix(h_A, 1 << 5, 1 << 5);
	InitMatrix(h_B, 1 << 5, 1 << 5);

	// transfer the data from CPU to GPU
	cudaMemcpy(d_A, h_A, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float)*N, cudaMemcpyHostToDevice);

	// add the matrix
	SumMatrixOnCPU(h_A, h_B, h_C, 1 << 5, 1 << 5);

	// sync
	SumMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, 1 << 5, 1 << 5);
	cudaDeviceSynchronize();

	// transfer the data from GPU to CPU
	cudaMemcpy(gpu_result, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

	PrintMatrix(h_C, 1 << 5, 1 << 5);
	PrintMatrix(gpu_result, 1 << 5, 1 << 5);

	// free the memory 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);
	free(gpu_result);
	return 0;
}