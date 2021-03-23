#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void GenerateMatrix(float *matrix, int nx, int ny)
{
	int i, j;
	float cnt = 0;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			matrix[i*nx + j] = cnt++;
		}
	}
	printf("[*] GenerateMatrix has done!\n");
}

void PrintMatrix(float *matrix, int nx, int ny)
{
	int i, j;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			printf("%.2f\t", matrix[i*nx + j]);
		}
		printf("\n");
	}
	printf("[*] PrintMatrix has done!\n");
}

/************************* matrix summary begin *************************/ 
void AddMatrixOnCPU(float *A, float *B, float *C, int nx, int ny)
{
	int i, j;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			C[i*nx + j] = A[i*nx + j] + B[i*nx + j];
		}
	}
	printf("[*] AddMatrix on CPU has done!\n");
}

__global__ void AddMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = i*nx + j;
	if (i <= nx && j <= ny)
	{
		C[idx] = A[idx] + B[idx];
	}
}
/************************* matrix summary done **************************/
//
//
//
/************************ matrix multiply begin *************************/
void MulMatrixOnCPU(float *A, float *B, float *C, int nx, int ny)
{
	int i, j, k;
	float sum = 0.0;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			sum = 0.0;
			for (k = 0; k < nx; k++)
			{
				sum = sum + A[i*nx + k] * B[k*nx + j];
			}
			C[i*nx + j] = sum;
		}
	}
}

__global__ void MulMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	if (i < nx && j < ny)   // we should to identify the "i" and "j" scope.
	{
		float sum = 0.0;
		for (k = 0; k < nx; k++)
		{
			sum += A[i*nx + k] * B[k*nx + j];
		}
		C[i*nx + j] = sum;
	}
}
/************************ matrix multiply end ***************************/

// compare the result
int Compare(float *cpu_ref, float *gpu_ref, int nx, int ny)
{
	int i, j;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			if (cpu_ref[i*nx + j] != gpu_ref[i*nx + j])
			{
				return 0;
			}
		}
	}
	return 1;
}


int main(int argc, char *argv[])
{
	LARGE_INTEGER begin_cpu, begin_gpu;
	LARGE_INTEGER end_cpu, end_gpu;
	LARGE_INTEGER freq_cpu, freq_gpu;
	
	// the size of the elements in the matrix can not be much larger....
	// because of my worse GPU: nVIDIA GeForce GT710
	unsigned int N = 1<<12; 
	int nx = (int)sqrt((float)N);
	int ny = (int)sqrt((float)N);

	float *A = NULL;
	float *B = NULL;
	float *C = NULL;
	float *gpu_ref = NULL;
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;

	// allocate the memory on CPU
	A = (float *)malloc(sizeof(float)* N);
	B = (float *)malloc(sizeof(float)* N);
	C = (float *)malloc(sizeof(float)* N);
	gpu_ref = (float *)malloc(sizeof(float)*N);
	// set the memory to zero
	memset(A, 0, sizeof(float)*N);
	memset(B, 0, sizeof(float)*N);
	memset(C, 0, sizeof(float)*N);
	memset(gpu_ref, 0, sizeof(float)*N);

	// allocate the memory on GPU
	cudaMalloc((float **)&d_A, sizeof(float)*N);
	cudaMalloc((float **)&d_B, sizeof(float)*N);
	cudaMalloc((float **)&d_C, sizeof(float)*N);
	// reset the memory to zero
	cudaMemset(d_A, 0, sizeof(float)*N);
	cudaMemset(d_B, 0, sizeof(float)*N);
	cudaMemset(d_C, 0, sizeof(float)*N);

	// generate the matrix on CPU
	GenerateMatrix(A, nx, ny);
	GenerateMatrix(B, nx, ny);

	// transfer the data from CPU to GPU
	cudaMemcpy(d_A, A, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(float)*N, cudaMemcpyHostToDevice);


	// set the grid number and the block thread number
	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// Add the matrix on CPU
	AddMatrixOnCPU(A, B, C, nx, ny);

	// Add the matrix on GPU
	AddMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();  // let the CPU wait the GPU to do its calculation.

	// transform the data from the GPU to CPU
	cudaMemcpy(gpu_ref, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

	if (Compare(C, gpu_ref, nx, ny))
	{
		printf("[*] Compare : Matrix_ADD => the result are the same!\n");
	}
	else
	{
		printf("[*] Compare : Matrix_ADD => the result are NOT the same...\n");
	}

	// begin to calculate the time consumption
	QueryPerformanceCounter(&freq_cpu);
	QueryPerformanceCounter(&begin_cpu);
	
	// test the matrix multiply
	MulMatrixOnCPU(A, B, C, nx, ny);
	// because of the GPU calculation use this function, so we should to make the same situation.
	cudaDeviceSynchronize();

	QueryPerformanceCounter(&end_cpu);
	printf("CPU time consumption:%f ms\n", 1000 * (float)(end_cpu.QuadPart - begin_cpu.QuadPart) / (float)freq_cpu.QuadPart);

	// begin to calculate the time consumption
	QueryPerformanceCounter(&freq_gpu);
	QueryPerformanceCounter(&begin_gpu);

	// test the matrix multiply on GPU
	MulMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();

	QueryPerformanceCounter(&end_gpu);
	printf("GPU time consumption:%f ms\n", 1000 * (float)(end_gpu.QuadPart - begin_gpu.QuadPart) / (float)freq_gpu.QuadPart);

	cudaMemcpy(gpu_ref, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// make the comparison
	if (Compare(C, gpu_ref, nx, ny))
	{
		printf("[*] Compare : Matrix_MUL => the result are the same!\n");
	}
	else
	{
		printf("[*] Compare : Matrix_MUL => the result are NOT the same...\n");
	}

	// Debug Print
	// PrintMatrix(gpu_ref, nx, ny);
	// PrintMatrix(C, nx, ny);
	
	free(A);
	free(B);
	free(C);
	free(gpu_ref);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}


// [*] GenerateMatrix has done!
// [*] GenerateMatrix has done!
// [*] AddMatrix on CPU has done!
// [*] Compare : Matrix_ADD = > the result are the same!
// [*] Compare : Matrix_MUL = > the result are the same!
// Press any key to continue...
