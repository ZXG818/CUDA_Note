#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// initialize the matrix
void initalMatrix(int *matrix, int nx, int ny)
{
	int i, j;
	int cnt = 0;
	for (j = 0; j < ny; j++)
	{
		for (i = 0; i < nx; i++)
		{
			matrix[cnt] = cnt;
			cnt++;
		}
	}
}

__global__ void AddMatrixOnGPU(int *A, int *B, int *C, int nx, int ny)
{
	int x, y;
	int idx;
	x = threadIdx.x + blockIdx.x * blockDim.x;
	y = threadIdx.y + blockIdx.y * blockDim.y;
	idx = y*nx + nx;
	if (x < nx && y < ny)
	{
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void Book(int *A, int *B, int *C, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	if (ix < nx)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			int idx = iy*nx + ix;
			C[idx] = A[idx] + B[idx];
		}
	}
	
}

void AddMatrixOnCPU(int *A, int *B, int *C, int nx, int ny)
{
	int i, j;
	int cnt = 0;
	for (j = 0; j < ny; j++)
	{
		for (i = 0; i < nx; i++)
		{
			C[cnt] = A[cnt] + B[cnt];
			cnt++;
		}
	}
}

void CheckResult(int *A, int *B, int nx, int ny)
{
	int i, j;
	int cnt = 0;
	for (j = 0; j < ny; j++)
	{
		for (i = 0; i < nx; i++)
		{
			if (abs(A[cnt] - B[cnt]) != 0)
			{
				printf("Do not match...\n");
				return;
			}
		}
	}
	printf("matched...\n");
}

int main(void)
{
	int nx = 1 << 10;
	int ny = 1 << 10;
	int nBytes = sizeof(int)*nx*ny;
	int *h_A, *h_B, *h_C, *gpuRef;
	int *d_A, *d_B, *d_C;
	h_A = (int *)malloc(nBytes);
	h_B = (int *)malloc(nBytes);
	h_C = (int *)malloc(nBytes);
	gpuRef = (int *)malloc(nBytes);

	initalMatrix(h_A, nx, ny);
	initalMatrix(h_B, nx, ny);

	AddMatrixOnCPU(h_A, h_B, h_C, nx, ny);

	cudaMalloc((int **)&d_A, nBytes);
	cudaMalloc((int **)&d_B, nBytes);
	cudaMalloc((int **)&d_C, nBytes);
	
	// memcpy from CPU TO GPU
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(32, 1);
	dim3 grid((nx + block.x - 1) / block.x, 1);
	//AddMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	Book << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();

	// copy data from GPU to CPU
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	CheckResult(h_C, gpuRef, nx, ny);

	// free the memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(gpuRef);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
