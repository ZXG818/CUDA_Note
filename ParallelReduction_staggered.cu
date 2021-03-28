#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// generate the array
void GenerateArray(int *A, int N)
{
  int i;
	for (i = 0; i < N; i++)
	{
		A[i] = i * 2;
	}
}

// get the summary on CPU
int SummaryOnCPU(int *A, int N)
{
	int sum = 0;
	int i;
	for (i = 0; i < N; i++)
	{
		sum += A[i];
	}
	return sum;
}


// get the summary on GPU
// use the neighboured summary 
// parameters: 
//            int *in_arr : the array on the current block
//            int *result : get the summary of the current block(in_arr)
//            int N       : the number of the elemenets in the current block.
__global__ void SummaryOnGPU_Neighboured(int *in_arr, int *result, int N)
{
	int stride = 1;
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int *local_arr = in_arr + blockIdx.x * blockDim.x; // restrict the threads on the current block.

	if (idx >= N)
	{
		return;
	}

	// begin to get the neighboured summary on GPU
	// caution: blockDim.x means the number of the threads which are in the current block.
	for (stride = 1; stride < blockDim.x; stride *= 2){
    if (tid % (stride * 2) == 0){
      local_arr[tid] += local_arr[tid + stride];
    }
    __syncthreads();  // let the threads synchronized in the current block.
	}
  if (tid == 0)
  {
    result[blockIdx.x] = local_arr[0];
  }
}


// use the staggered parallel reduction
__global__ void SummaryOnGPU_Staggered(int *in_arr, int *result, int N)
{
  int stride = 0;
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int *local_arr = in_arr + blockIdx.x * blockDim.x; // restrict the threads on the current block.

  if (idx >= N)
    return;

  for (stride = blockDim.x / 2; stride; stride /= 2){
    if (tid < stride){
      local_arr[tid] += local_arr[tid + stride];
    }
    __syncthreads();
  }
  
  if (tid == 0)
  {
    result[blockIdx.x] = local_arr[0];
  }
}

// main function
int main(int argc, char *argv[])
{
  int N = 1 << 10;
  int i;
  int final_result = 0;

  int *h_A = NULL;
  int *h_result = 0;
  int *d_A = NULL;
  int *d_result = NULL;
  
  // allocate the memory on CPU
  h_A = (int *)malloc(sizeof(int)*N);
  memset(h_A, 0, sizeof(int)*N);

  // allocate the memory on GPU
  cudaMalloc((int **)&d_A, sizeof(int)*N);
  cudaMemset(d_A, 0, sizeof(int)*N);

  // generate the array
  GenerateArray(h_A, N);
  cudaMemcpy(d_A, h_A, sizeof(int)*N, cudaMemcpyHostToDevice);

  // define the <<<grid, block>>>
  dim3 block(256, 1);
  dim3 grid((N + block.x - 1) / block.x, 1);

  h_result = (int *)malloc(sizeof(int)*grid.x);
  memset(h_result, 0, sizeof(int)*grid.x);
  cudaMalloc((int **)&d_result, sizeof(int)*grid.x);
  cudaMemset(d_result, 0, sizeof(int)*grid.x);

  // calculate the array on GPU with the neighboured method.
  SummaryOnGPU_Neighboured << <grid, block>> >(d_A, d_result, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_result, d_result, sizeof(int)*grid.x, cudaMemcpyDeviceToHost);
  for (i = 0; i < grid.x; i++){
    final_result += h_result[i];
  }
  printf("The summary on GPU with neighboured style is : %d\n", final_result);

  // reset the value
  cudaMemcpy(d_A, h_A, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(int)*grid.x);
  memset(h_result, 0, sizeof(int)*grid.x);
  final_result = 0;

  // calculate the array on GPU with the staggered method.
  SummaryOnGPU_Staggered << <grid, block >> >(d_A, d_result, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_result, d_result, sizeof(int)*grid.x, cudaMemcpyDeviceToHost);
  for (i = 0; i < grid.x; i++){
    final_result += h_result[i];
  }
  printf("The summary on GPU with staggered style is : %d\n", final_result);

  // calculate the array on CPU
  printf("The summary on CPU is : %d\n", SummaryOnCPU(h_A, N));

  cudaFree(d_A);
  cudaFree(d_result);
  free(h_A);
  free(h_result);
  return 0;
}
// result:
// The summary on GPU with neighboured style is : 1047552
// The summary on GPU with staggered style is : 1047552
// The summary on CPU is : 1047552
