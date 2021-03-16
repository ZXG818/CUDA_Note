#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void checkResult(float *A, float *B, const int nx, const int ny)
{
	int i = 0;
	int j = 0;
	int cnt = 0;
	double err = 1.0E-6;
	for (j = 0; j < ny; j++)
	{
		for (i = 0; i < nx; i++)
		{
			if (fabs(A[cnt] - B[cnt]) > err)
			{
				printf("Do not match...\n");
				return;
			}
			cnt++;
		}
	}
	printf("matched!\n");
}

void initialData(float *a, int nx, int ny)
{
	int i = nx;
	int j = ny;
	int cnt = 0;
	for (j = 0; j < ny; j++)
	{
		for (i = 0; i < nx; i++)
		{
			a[cnt] = cnt;
			cnt++;
		}
	}
}

// summary matrix on CPU
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	int i = 0;
	int j = 0;
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

void PrintMatrix(float *a)
{
	int i;
	for (i = 0; i < 10; i++)
	{
		printf("%f  ", a[i]);
	}
	printf("\n");
}

// summary matrix on GPU
__global__ void sumMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = y*nx + x;
	if (x < nx && y < ny)
	{
		C[idx] = A[idx] + B[idx];
	}
}


__global__ void test()
{
	printf("hello\n");
}

int main(int argc, char *argv[])
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	cudaSetDevice(dev);
	int nx = 1 << 10;    // 此处对显卡的限制比较明显，书中可以让nx和ny分别为1<<14，所以nx*ny = 1<<28，但是我的显卡不行。
	int ny = 1 << 10;
	int nxy = nx * ny;
	int nBytes = sizeof(float)*nxy;
	printf("Matrix size: nx:%d, ny:%d\n", nx, ny);

	float *h_A, *h_B, *h_C, *gpuRef;
	float *d_A, *d_B, *d_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);
	memset(gpuRef, 0, nBytes);

	cudaMalloc((void **)&d_A, nBytes);
	cudaMalloc((void **)&d_B, nBytes);
	cudaMalloc((void **)&d_C, nBytes);

	// initialize the data
	initialData(h_A, nx, ny);
	initialData(h_B, nx, ny);

	// copy the data from CPU to GPU
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	// call the summary function
	sumMatrixOnHost(h_A, h_B, h_C, nx, ny);

	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	sumMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();

	// copy the data from GPU to CPU
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	// check the result
	checkResult(h_C, gpuRef, nx, ny);

	PrintMatrix(h_C);
	PrintMatrix(gpuRef);

	// free the memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(gpuRef);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	return 0;
}