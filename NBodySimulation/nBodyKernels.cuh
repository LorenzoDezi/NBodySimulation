#include "cuda_runtime.h"
#include <stdio.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "constants.h"

//Naive kernel to update positions and accelerations for nBody simulations
__global__ void updateSimple(float4 * positions, float deltaTime);

//Returns an updated position given its acceleration and the deltaTime
__device__ float updatedPosition(float pos, float acc, float deltaTime);

__global__ void generatePointInsideSphere(float4 * points, curandState * states);

