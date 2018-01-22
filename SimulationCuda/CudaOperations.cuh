#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <math.h>

__host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ int3 operator/(int3 a, int b)
{
	return make_int3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


__host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


__host__ __device__ float euclideanDistance(float3 v)
{
	return sqrtf(dot(v, v));
}