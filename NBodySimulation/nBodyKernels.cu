#include "nBodyKernels.cuh"
#include <cmath>



__global__ void updateSimple(float4 * positions, float4 * velocities)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float position_i_x = positions[i].x; //the fourth value of position is the scale
	float position_i_y = positions[i].y;
	float position_i_z = positions[i].z;
	float position_i_w = positions[i].w;
	float velocity_i_x = velocities[i].x;
	float velocity_i_y = velocities[i].y;
	float velocity_i_z = velocities[i].z;
	float position_j_x, position_j_y, position_j_z, position_j_w;
	float newAcc_i_x = 0, newAcc_i_y = 0, newAcc_i_z = 0;
	float dist_ij_x, dist_ij_y, dist_ij_z;
	for (int j = 0; j < N; j++) {
		position_j_x = positions[j].x;
		position_j_y = positions[j].y;
		position_j_z = positions[j].z;
		position_j_w = positions[j].w;
		dist_ij_x = position_j_x - position_i_x;
		dist_ij_y = position_j_y - position_i_y;
		dist_ij_z = position_j_z - position_i_z;		
		float distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
		float denom = (sqrtf(distSqr * distSqr * distSqr));
		float mass_jOverDenom = position_j_w * MASS_MULTIPLIER / denom;
		newAcc_i_x += mass_jOverDenom * dist_ij_x;
		newAcc_i_y += mass_jOverDenom * dist_ij_y;
		newAcc_i_z += mass_jOverDenom * dist_ij_z;
	};
	newAcc_i_x *= G;
	newAcc_i_y *= G;
	newAcc_i_z *= G;
	//Integration step
	velocity_i_x += newAcc_i_x * TIME_STEP;
	velocity_i_y += newAcc_i_y * TIME_STEP;
	velocity_i_z += newAcc_i_z * TIME_STEP;
	positions[i] = { 
		position_i_x + velocity_i_x * TIME_STEP,
		position_i_y + velocity_i_y * TIME_STEP,
		position_i_z + velocity_i_z * TIME_STEP,
		position_i_w
	};
	velocities[i] = { velocity_i_x, velocity_i_y, velocity_i_z, 0.f };
}

__global__ void updateSimpleLoopUnroll(float4 * positions, float4 * velocities)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float position_i_x = positions[i].x;
	float position_i_y = positions[i].y;
	float position_i_z = positions[i].z;
	float position_i_w = positions[i].w;
	float velocity_i_x = velocities[i].x;
	float velocity_i_y = velocities[i].y;
	float velocity_i_z = velocities[i].z;
	float position_j_x, position_j_y, position_j_z, position_j_w;
	float newAcc_i_x = 0, newAcc_i_y = 0, newAcc_i_z = 0;
	float dist_ij_x, dist_ij_y, dist_ij_z, distSqr, denom, mass_jOverDenom;
	for (int j = 0; j < N; j+=2) {
		position_j_x = positions[j].x;
		position_j_y = positions[j].y;
		position_j_z = positions[j].z;
		position_j_w = positions[j].w;
		dist_ij_x = position_j_x - position_i_x;
		dist_ij_y = position_j_y - position_i_y;
		dist_ij_z = position_j_z - position_i_z;
		distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
		denom = (sqrtf(distSqr * distSqr * distSqr));
		mass_jOverDenom = position_j_w * MASS_MULTIPLIER / denom;
		newAcc_i_x += mass_jOverDenom * dist_ij_x;
		newAcc_i_y += mass_jOverDenom * dist_ij_y;
		newAcc_i_z += mass_jOverDenom * dist_ij_z;
		position_j_x = positions[j+1].x;
		position_j_y = positions[j+1].y;
		position_j_z = positions[j+1].z;
		position_j_w = positions[j+1].w;
		dist_ij_x = position_j_x - position_i_x;
		dist_ij_y = position_j_y - position_i_y;
		dist_ij_z = position_j_z - position_i_z;
		distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
		denom = (sqrtf(distSqr * distSqr * distSqr));
		mass_jOverDenom = position_j_w * MASS_MULTIPLIER / denom;
		newAcc_i_x += mass_jOverDenom * dist_ij_x;
		newAcc_i_y += mass_jOverDenom * dist_ij_y;
		newAcc_i_z += mass_jOverDenom * dist_ij_z;
	};
	newAcc_i_x *= G;
	newAcc_i_y *= G;
	newAcc_i_z *= G;
	//Integration step
	velocity_i_x += newAcc_i_x * TIME_STEP;
	velocity_i_y += newAcc_i_y * TIME_STEP;
	velocity_i_z += newAcc_i_z * TIME_STEP;
	positions[i] = {
		position_i_x + velocity_i_x * TIME_STEP,
		position_i_y + velocity_i_y * TIME_STEP,
		position_i_z + velocity_i_z * TIME_STEP,
		position_i_w
	};
	velocities[i] = { velocity_i_x, velocity_i_y, velocity_i_z, 0.f };
}

__global__ void updateShared(float4 * positions, float4 * velocities)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) return;
	
	__shared__ float4 sharedPositions[BLOCK_DIM];

	float position_i_x = positions[i].x; //the fourth value of position is the scale
	float position_i_y = positions[i].y;
	float position_i_z = positions[i].z;
	float position_i_w = positions[i].w;
	float velocity_i_x = velocities[i].x;
	float velocity_i_y = velocities[i].y;
	float velocity_i_z = velocities[i].z;
	float newAcc_i_x = 0, newAcc_i_y = 0, newAcc_i_z = 0;
	float dist_ij_x, dist_ij_y, dist_ij_z, distSqr, denom, mass_jOverDenom;
	int block_i, size_curr;
	int n_blocks = (N + BLOCK_DIM) / BLOCK_DIM;
	for (int i = 0; i < n_blocks; i++) {
		block_i = i * BLOCK_DIM;
		if (block_i + threadIdx.x < N) {
			sharedPositions[threadIdx.x] = positions[block_i + threadIdx.x];
		}
		__syncthreads();
		size_curr = i == n_blocks - 1 ? N - i * BLOCK_DIM : BLOCK_DIM;
		for (int j = 0; j < size_curr; j++) {
			dist_ij_x = sharedPositions[j].x - position_i_x;
			dist_ij_y = sharedPositions[j].y - position_i_y;
			dist_ij_z = sharedPositions[j].z - position_i_z;
			distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
			mass_jOverDenom = sharedPositions[j].w * MASS_MULTIPLIER / (sqrtf(distSqr * distSqr * distSqr));
			newAcc_i_x += mass_jOverDenom * dist_ij_x;
			newAcc_i_y += mass_jOverDenom * dist_ij_y;
			newAcc_i_z += mass_jOverDenom * dist_ij_z;
		}
		__syncthreads();
	}
	newAcc_i_x *= G;
	newAcc_i_y *= G;
	newAcc_i_z *= G;
	//Integration step
	velocity_i_x += newAcc_i_x * TIME_STEP;
	velocity_i_y += newAcc_i_y * TIME_STEP;
	velocity_i_z += newAcc_i_z * TIME_STEP;
	positions[i] = {
		position_i_x + velocity_i_x * TIME_STEP,
		position_i_y + velocity_i_y * TIME_STEP,
		position_i_z + velocity_i_z * TIME_STEP,
		position_i_w
	};
	velocities[i] = { velocity_i_x, velocity_i_y, velocity_i_z, 0.f };
}

__global__ void updateSharedLoopUnroll(float4 * positions, float4 * velocities)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) return;

	__shared__ float4 sharedPositions[BLOCK_DIM];

	float position_i_x = positions[i].x; //the fourth value of position is the scale
	float position_i_y = positions[i].y;
	float position_i_z = positions[i].z;
	float position_i_w = positions[i].w;
	float velocity_i_x = velocities[i].x;
	float velocity_i_y = velocities[i].y;
	float velocity_i_z = velocities[i].z;
	float newAcc_i_x = 0, newAcc_i_y = 0, newAcc_i_z = 0;
	float dist_ij_x, dist_ij_y, dist_ij_z, distSqr, denom, mass_jOverDenom;
	int block_i, size_curr;
	int n_blocks = (N + BLOCK_DIM) / BLOCK_DIM;
	for (int i = 0; i < n_blocks; i++) {
		block_i = i * BLOCK_DIM;
		if (block_i + threadIdx.x < N) {
			sharedPositions[threadIdx.x] = positions[block_i + threadIdx.x];
		}
		__syncthreads();
		size_curr = i == n_blocks - 1 ? N - i * BLOCK_DIM : BLOCK_DIM;
		for (int j = 0; j < size_curr/2; j+=2) {
			dist_ij_x = sharedPositions[j].x - position_i_x;
			dist_ij_y = sharedPositions[j].y - position_i_y;
			dist_ij_z = sharedPositions[j].z - position_i_z;
			distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
			mass_jOverDenom = sharedPositions[j].w * MASS_MULTIPLIER / (sqrtf(distSqr * distSqr * distSqr));
			newAcc_i_x += mass_jOverDenom * dist_ij_x;
			newAcc_i_y += mass_jOverDenom * dist_ij_y;
			newAcc_i_z += mass_jOverDenom * dist_ij_z;
			dist_ij_x = sharedPositions[j+1].x - position_i_x;
			dist_ij_y = sharedPositions[j+1].y - position_i_y;
			dist_ij_z = sharedPositions[j+1].z - position_i_z;
			distSqr = dist_ij_x * dist_ij_x + dist_ij_y * dist_ij_y + dist_ij_z * dist_ij_z + EPS_SQUARED;
			mass_jOverDenom = sharedPositions[j+1].w * MASS_MULTIPLIER / (sqrtf(distSqr * distSqr * distSqr));
			newAcc_i_x += mass_jOverDenom * dist_ij_x;
			newAcc_i_y += mass_jOverDenom * dist_ij_y;
			newAcc_i_z += mass_jOverDenom * dist_ij_z;
		}
		__syncthreads();
	}
	newAcc_i_x *= G;
	newAcc_i_y *= G;
	newAcc_i_z *= G;
	//Integration step
	velocity_i_x += newAcc_i_x * TIME_STEP;
	velocity_i_y += newAcc_i_y * TIME_STEP;
	velocity_i_z += newAcc_i_z * TIME_STEP;
	positions[i] = {
		position_i_x + velocity_i_x * TIME_STEP,
		position_i_y + velocity_i_y * TIME_STEP,
		position_i_z + velocity_i_z * TIME_STEP,
		position_i_w
	};
	velocities[i] = { velocity_i_x, velocity_i_y, velocity_i_z, 0.f };
}

__global__ void generatePointInsideSphere(float4 * points, curandState * states)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= N) return;
	curand_init(tid, tid, tid, &states[tid]);
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
