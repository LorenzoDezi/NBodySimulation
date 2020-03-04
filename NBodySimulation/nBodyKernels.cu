#include "nBodyKernels.cuh"
#include <cmath>



__global__ void updateSimple(float4 * positions, float4 * velocities)
{
	//DEBUG
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float4 position_i = positions[i]; //the fourth value of position is the scale
	float4 velocity_i = velocities[i];
	float4 position_j;
	float4 new_acc = float4();
	for (int j = 0; j < N; j++) {
		position_j = positions[j];
		float3 dist_ij = { position_j.x - position_i.x, position_j.y - position_i.y, position_j.z - position_i.z };
		float distSqr = dist_ij.x * dist_ij.x + dist_ij.y * dist_ij.y + dist_ij.z * dist_ij.z + EPS_SQUARED;
		float mass_jOverDenom = position_j.w * MASS_MULTIPLIER / (sqrtf(distSqr * distSqr * distSqr));
		new_acc.x += mass_jOverDenom * dist_ij.x;
		new_acc.y += mass_jOverDenom * dist_ij.y;
		new_acc.z += mass_jOverDenom * dist_ij.z;
	};
	new_acc.x *= G;
	new_acc.y *= G;
	new_acc.z *= G;
	//Integration step
	velocity_i.x += new_acc.x * TIME_STEP;
	velocity_i.y += new_acc.y * TIME_STEP;
	velocity_i.z += new_acc.z * TIME_STEP;
	positions[i] = { 
		position_i.x + velocity_i.x * TIME_STEP,
		position_i.y + velocity_i.y * TIME_STEP,
		position_i.z + velocity_i.z * TIME_STEP,
		position_i.w
	};
	velocities[i] = velocity_i;
}

__global__ void generatePointInsideSphere(float4 * points, curandState * states)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= N) return;
	curand_init(tid, 5000, 500, &states[tid]);
	float x, y, z, w, trial = 0;
	do {
		x = (curand_uniform(&states[tid]) * 2 - 1) * RADIUS * 5;
		y = (curand_uniform(&states[tid]) * 2 - 1) * RADIUS * 5;
		z = (curand_uniform(&states[tid]) * 2 - 1) * RADIUS * 5;
		w = curand_uniform(&states[tid]) * SCALE;
		trial += 1;
	} while (x * x + y * y + z * z > RADIUS && trial <= MAX_TRIALS);
	points[tid] = { x, y, z, w };
	printf("Index: %d, x: %.3f, y: %.3f, z: %.3f, mass: %.3f \n", tid, x, y, z, w);
}
