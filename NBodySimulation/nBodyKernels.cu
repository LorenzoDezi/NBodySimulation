#include "nBodyKernels.cuh"

__global__ void updateSimple(float4 * accelerations, float4 * positions)
{
	//DEBUG
	int tid = threadIdx.x;
	float val = positions[tid].x;
	printf("Value: %.3f", val);
}