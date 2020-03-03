#include "cuda_runtime.h"
#include <stdio.h>
#include "device_launch_parameters.h"

//Naive kernel to update positions and accelerations for nBody simulations
__global__ void updateSimple(float4 * accelerations, float4 * positions, float deltaTime);

//Returns an updated position given its acceleration and the deltaTime
__device__ float updatedPosition(float pos, float acc, float deltaTime);

