#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>  // use the QPC 

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
inline void AddMatrixOnCPU(float *A, float *B, float *C, int nx, int ny)
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

__global__ inline void AddMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
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
inline void MulMatrixOnCPU(float *A, float *B, float *C, int nx, int ny)
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

__global__ inline void MulMatrixOnGPU(float *A, float *B, float *C, int nx, int ny)
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
	unsigned int N = 1 << 12;
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

	QueryPerformanceCounter(&end_cpu);
	printf("CPU time consumption:%f ms\n", 1000*(float)(end_cpu.QuadPart - begin_cpu.QuadPart) / (float)freq_cpu.QuadPart);

	// test the matrix multiply on GPU
	MulMatrixOnGPU << <grid, block >> >(d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();

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


// nvprof check
// C:\Users\HP\Desktop\test\x64\Debug > nvprof test.exe
// 	== 18712 == NVPROF is profiling process 18712, command: test.exe
// 	[*] GenerateMatrix has done!
// 	[*] GenerateMatrix has done!
// 	[*] AddMatrix on CPU has done!
// 	[*] Compare : Matrix_ADD = > the result are the same!
// 	CPU time consumption : 0.000002 ms
// 	GPU time consumption : 0.000002 ms
// 	[*] Compare : Matrix_MUL = > the result are the same!
// 	== 18712 == Profiling application : test.exe
// 	== 18712 == Profiling result :
//   Type  Time(%)      Time     Calls       Avg       Min       Max  Name
// 	  GPU activities : 91.91%  718.66us         1  718.66us  718.66us  718.66us  MulMatrixOnGPU(float*, float*, float*, int, int)
// 	  3.62%  28.285us         1  28.285us  28.285us  28.285us  AddMatrixOnGPU(float*, float*, float*, int, int)
// 	  1.93%  15.071us         3  5.0230us  3.8390us  7.3600us[CUDA memset]
// 	  1.28%  10.047us         2  5.0230us  4.9280us  5.1190us[CUDA memcpy DtoH]
// 	  1.26%  9.8870us         2  4.9430us  4.5760us  5.3110us[CUDA memcpy HtoD]
// 	  API calls : 90.76%  331.25ms         3  110.42ms  2.6000us  331.25ms  cudaMalloc
// 	  8.46%  30.874ms         1  30.874ms  30.874ms  30.874ms  cuDevicePrimaryCtxRelease
// 	  0.24%  871.50us         4  217.88us  55.900us  641.20us  cudaMemcpy
// 	  0.24%  870.40us         3  290.13us  12.400us  790.50us  cudaDeviceSynchronize
// 	  0.17%  616.90us         1  616.90us  616.90us  616.90us  cuModuleUnload
// 	  0.07%  242.00us        97  2.4940us     100ns  127.40us  cuDeviceGetAttribute
// 	  0.04%  149.10us         3  49.700us  6.6000us  122.20us  cudaFree
// 	  0.01%  47.200us         2  23.600us  15.100us  32.100us  cudaLaunchKernel
// 	  0.01%  22.300us         1  22.300us  22.300us  22.300us  cuDeviceTotalMem
// 	  0.00%  14.100us         3  4.7000us  1.4000us  10.600us  cudaMemset
// 	  0.00%  6.8000us         1  6.8000us  6.8000us  6.8000us  cuDeviceGetPCIBusId
// 	  0.00%  2.7000us         3     900ns     200ns  2.3000us  cuDeviceGetCount
// 	  0.00%  1.5000us         2     750ns     100ns  1.4000us  cuDeviceGet
// 	  0.00 % 800ns         1     800ns     800ns     800ns  cuDeviceGetName
// 	  0.00 % 400ns         1     400ns     400ns     400ns  cuDeviceGetUuid
// 	  0.00 % 200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
// 
//   C : \Users\HP\Desktop\test\x64\Debug > cd ..
// 
//   C:\Users\HP\Desktop\test\x64 > cd Release
// 
// C : \Users\HP\Desktop\test\x64\Release > nvprof test.exe
// 	== 18808 == NVPROF is profiling process 18808, command: test.exe
// 	[*] GenerateMatrix has done!
// 	[*] GenerateMatrix has done!
// 	[*] AddMatrix on CPU has done!
// 	[*] Compare : Matrix_ADD = > the result are the same!
// 	CPU time consumption : 0.000000 ms
// 	[*] Compare : Matrix_MUL = > the result are the same!
// 	== 18808 == Profiling application : test.exe
// 	== 18808 == Profiling result :
//   Type  Time(%)      Time     Calls       Avg       Min       Max  Name
// 	  GPU activities : 91.07%  599.83us         1  599.83us  599.83us  599.83us  MulMatrixOnGPU(float*, float*, float*, int, int)
// 	  3.82%  25.150us         1  25.150us  25.150us  25.150us  AddMatrixOnGPU(float*, float*, float*, int, int)
// 	  1.97%  12.991us         3  4.3300us  3.6790us  5.6320us[CUDA memset]
// 	  1.61%  10.624us         2  5.3120us  5.3120us  5.3120us[CUDA memcpy HtoD]
// 	  1.53%  10.079us         2  5.0390us  4.8000us  5.2790us[CUDA memcpy DtoH]
// 	  API calls : 73.36%  96.757ms         3  32.252ms  3.1000us  96.746ms  cudaMalloc
// 	  25.46%  33.576ms         1  33.576ms  33.576ms  33.576ms  cuDevicePrimaryCtxRelease
// 	  0.52%  691.50us         2  345.75us  59.600us  631.90us  cudaDeviceSynchronize
// 	  0.17%  224.60us         4  56.150us  25.500us  81.700us  cudaMemcpy
// 	  0.16%  213.70us         1  213.70us  213.70us  213.70us  cuModuleUnload
// 	  0.13%  175.10us         3  58.366us  6.4000us  152.30us  cudaFree
// 	  0.12%  157.10us        97  1.6190us     100ns  69.500us  cuDeviceGetAttribute
// 	  0.03%  42.400us         2  21.200us  13.300us  29.100us  cudaLaunchKernel
// 	  0.02%  24.400us         1  24.400us  24.400us  24.400us  cuDeviceTotalMem
// 	  0.01%  15.300us         3  5.1000us  1.5000us  11.900us  cudaMemset
// 	  0.00%  6.5000us         1  6.5000us  6.5000us  6.5000us  cuDeviceGetPCIBusId
// 	  0.00%  2.6000us         3     866ns     200ns  2.2000us  cuDeviceGetCountt
// 	  0.00%  1.4000us         2     700ns     100ns  1.3000us  cuDeviceGet
// 	  0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
// 	  0.00 % 400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
// 	  0.00 % 300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
