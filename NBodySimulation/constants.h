#pragma once

#define N 10000
#define BLOCK_DIM 256
#define RADIUS 100
#define G 9.81
//DEBUG
#define EPS 2
//The fourth component of the position vector is the scale of the object.
//The mass multiplier * scale will define its mass
#define MASS_MULTIPLIER 1000
#define SCALE 5
#define TIME_STEP 1
#define MAX_TRIALS 1000