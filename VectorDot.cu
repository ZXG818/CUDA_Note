#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREAD_PER_BLOCK 32

void GenerateVector(int *v, int N)
{
  int i;
  for (i = 0; i < N; i++){
    v[i] = i;
  }
}

int MulVectorOnCPU(int *u, int *v, int N)
{
  int i;
  int sum = 0.0;
  for (i = 0; i < N; i++){
    sum += u[i] * v[i];
  }
  return sum;
}

// use the staggered coupled method.
__global__ void MulVectorOnGPU(int *u, int*v, int *w, int N)
{
  int tid = threadIdx.x;
  int idx = tid + blockIdx.x * blockDim.x;

  int *local_u = u + blockIdx.x * blockDim.x;
  int *local_v = v + blockIdx.x * blockDim.x;
  int stride = 0;

  if (idx >= N){
    return;
  }

  // multiply each element in the vector
  local_u[tid] = local_u[tid] * local_v[tid];
  __syncthreads();

  // summary reduction
  for (stride = blockDim.x / 2; stride; stride /= 2){
    if (tid < stride){
      local_u[tid] += local_u[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0){
    w[blockIdx.x] = local_u[0];
  }
}

int main(int argc, char *argv[])
{
  int N = 1 << 10;
  int *h_u = NULL;
  int *h_v = NULL;
  int *gpu_ref = NULL;
  int *d_u = NULL;
  int *d_v = NULL;
  int *d_w = NULL;

  int i;
  int sum = 0.0;
  
  h_u = (int *)malloc(sizeof(int)*N);
  memset(h_u, 0, sizeof(int)*N);
  h_v = (int *)malloc(sizeof(int)*N);
  memset(h_v, 0, sizeof(int)*N);

  cudaMalloc((int**)&d_u, sizeof(int)*N);
  cudaMemset(d_u, 0, sizeof(int)*N);
  cudaMalloc((int**)&d_v, sizeof(int)*N);
  cudaMemset(d_v, 0, sizeof(int)*N);

  // define the grid number and the block number
  dim3 block(THREAD_PER_BLOCK, 1);
  dim3 grid((N + block.x - 1) / block.x, 1);

  gpu_ref = (int*)malloc(sizeof(int)*grid.x);
  memset(gpu_ref, 0, sizeof(int)*grid.x);
  cudaMalloc((int**)&d_w, sizeof(int)*grid.x);
  cudaMemset(d_w, 0, sizeof(int)*grid.x);

  GenerateVector(h_u, N);
  GenerateVector(h_v, N);

  // transfer the data from CPU to GPU
  cudaMemcpy(d_u, h_u, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, sizeof(int)*N, cudaMemcpyHostToDevice);

  // CPU calculation
  printf("MulVectorOnCPU is : %d\n", MulVectorOnCPU(h_u, h_v, N));

  // GPU calculation
  MulVectorOnGPU << <grid, block >> >(d_u, d_v, d_w, N);
  cudaDeviceSynchronize();
  cudaMemcpy(gpu_ref, d_w, sizeof(int)*grid.x, cudaMemcpyDeviceToHost);
  for (i = 0; i < grid.x; i++){
    sum += gpu_ref[i];
    printf("%d\n", gpu_ref[i]);
  }
  
  printf("MulVectorOnGPU is : %d\n", sum);

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);

  free(h_u);
  free(h_v);
  free(gpu_ref);

  return 0;
}

// results:
// MulVectorOnCPU is : 357389824
// 10416
// 74928
// 204976
// 400560
// 661680
// 988336
// 1380528
// 1838256
// 2361520
// 2950320
// 3604656
// 4324528
// 5109936
// 5960880
// 6877360
// 7859376
// 8906928
// 10020016
// 11198640
// 12442800
// 13752496
// 15127728
// 16568496
// 18074800
// 19646640
// 21284016
// 22986928
// 24755376
// 26589360
// 28488880
// 30453936
// 32484528
// MulVectorOnGPU is : 357389824
// Press any key to continue...
