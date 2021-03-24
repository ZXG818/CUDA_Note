#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <windows.h>

#define CHECK(call) \
do{\
const cudaError_t error = call;\
if (error != cudaSuccess)\
{\
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1);\
}\
} while (0)


// record the time
double cpuSecond()
{
	return 0;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0e-8;
	int match = 1;
	for (int i = 0; i < N; i++)
	{
		if (fabs(hostRef[i] - gpuRef[i])>epsilon)
		{
			match = 0;
			printf("Arrays do not match...\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match)
		printf("Arrays match...\n");
	return;
}

void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
	}
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) C[i] = A[i] + B[i];
}


int main(int argc, char *argv[])
{
	LARGE_INTEGER begin, begin_cpu;
	LARGE_INTEGER end, end_cpu;
	LARGE_INTEGER freq, freq_cpu;
	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 1<<24;
	printf("Vector size %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	float *d_A, *d_B, *d_C;
	cudaMalloc((float **)&d_A, nBytes);
	cudaMalloc((float **)&d_B, nBytes);
	cudaMalloc((float **)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(1024);
	dim3 grid((nElem + block.x - 1) / block.x);
	
	QueryPerformanceCounter(&freq);
	QueryPerformanceCounter(&begin);
	sumArrayOnGPU << <grid, block >> >(d_A, d_B, d_C, nElem);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&end);
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
	printf("GPU time consumption:%f ms\n", 1000 * (float)(end.QuadPart - begin.QuadPart) / (float)freq.QuadPart);

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	QueryPerformanceCounter(&freq_cpu);
	QueryPerformanceCounter(&begin_cpu);
	sumArrayOnHost(h_A, h_B, hostRef, nElem);
	QueryPerformanceCounter(&end_cpu);
	printf("CPU time consumption:%f ms\n", 1000 * (float)(end_cpu.QuadPart - begin_cpu.QuadPart) / (float)freq_cpu.QuadPart);

	checkResult(hostRef, gpuRef, nElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return 0;
}
