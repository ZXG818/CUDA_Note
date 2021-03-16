#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void helloFromGPU(void)
{
	if (threadIdx.x == 5)
	{
		printf("hello from GPU%d!\n", threadIdx.x);
	}
}


int main(void)
{
	printf("Hello world from CPU!\n");
	helloFromGPU << <1, 10 >> > ();
	//cudaDeviceReset();
	cudaDeviceSynchronize();
	return 0;
}