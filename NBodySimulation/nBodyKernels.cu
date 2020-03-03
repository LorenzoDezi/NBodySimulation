#include "nBodyKernels.cuh"
#include <cmath>

#define G 9.81
//DEBUG
#define N 1000
#define EPS 5
//The fourth component of the position vector is the scale of the object.
//The mass multiplier * scale will define its mass
#define MASS_MULTIPLIER 5
#define TIME_STEP 1

__global__ void updateSimple(float4 * accelerations, float4 * positions, float deltaTime)
{
	//DEBUG
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float4 position_i = positions[i]; //the fourth value of position is the mass
	float4 position_j;
	float4 new_acc = float4();
	for (int j = 0; j < N; j++) {
		position_j = positions[j];
		float3 dist_ij = { position_j.x - position_i.x, position_j.y - position_i.y, position_j.z - position_i.z };
		float distSqr = dist_ij.x * dist_ij.x + dist_ij.y * dist_ij.y + dist_ij.z * dist_ij.z + EPS * EPS;
		float denom = sqrtf(distSqr * distSqr * distSqr);
		new_acc.x += (position_j.w * MASS_MULTIPLIER * dist_ij.x) / denom;
		new_acc.y += (position_j.w * MASS_MULTIPLIER * dist_ij.y) / denom;
		new_acc.z += (position_j.w * MASS_MULTIPLIER * dist_ij.z) / denom;
	};
	new_acc.x *= G;
	new_acc.y *= G;
	new_acc.z *= G;
	positions[i] = { 
		updatedPosition(position_i.x, new_acc.x, deltaTime),
		updatedPosition(position_i.y, new_acc.y, deltaTime), 
		updatedPosition(position_i.z, new_acc.z, deltaTime),
		position_i.w
	};
	//DEBUG
	if (i == 0) {
		printf("Updated Position: %.3f \n", updatedPosition(position_i.x, new_acc.x, deltaTime));
		printf("Position: %.3f \n", positions[i].x);
		printf("Acceleration: %.3f \n", new_acc.x);
	}
}

__device__ float updatedPosition(float pos, float acc, float deltaTime) {
	return pos + ((1 / 2.f) * acc * deltaTime * deltaTime);
}