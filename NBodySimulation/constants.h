#pragma once

#define N 20000
#define BLOCK_DIM 512
#define RADIUS 100
#define G 9.81f
//Softening factor
#define EPS_SQUARED 100.f
//The fourth component of the position vector is the scale of the object.
//The mass multiplier * scale will define its mass
#define MASS_MULTIPLIER 50
#define SCALE 5
#define TIME_STEP 0.05f
#define MAX_TRIALS 1000