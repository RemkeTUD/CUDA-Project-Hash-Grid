#pragma once

#include "Loader.h"
#include "CudaOperations.cuh"
#define BLOCKSIZE 128

#define HANDLE_ERROR(err)\
	(handleCudaError(err, __FILE__, __LINE__))

struct Particle {
	float3 pos;
	float radius;
};

static void handleCudaError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


/* Calculates the grid position */
__host__ __device__ int3 calcGridPos(float3 p, PSystemInfo pSysInfo)
{
	int3 gridPos;
	gridPos.x = floor((p.x - pSysInfo.worldOrigin.x) / pSysInfo.cellSize.x);
	gridPos.y = floor((p.y - pSysInfo.worldOrigin.y) / pSysInfo.cellSize.y);
	gridPos.z = floor((p.z - pSysInfo.worldOrigin.z) / pSysInfo.cellSize.z);
	return gridPos;
}

/* Calculates the cell hash value */
__host__ __device__ uint calcGridHash(int3 gridPos, PSystemInfo pSysInfo)
{
	return gridPos.z * pSysInfo.gridSize.y * pSysInfo.gridSize.x + gridPos.y * pSysInfo.gridSize.x + gridPos.x;
}

/* Calculates grid position with a given hash */
__host__ __device__ int3 calcGridPosFromHash(uint hash, int3 gridSize)
{
	int3 pos;
	uint areaXY = gridSize.x * gridSize.y;
	pos.x = hash % gridSize.x;
	pos.y = ((hash - pos.x) % (areaXY)) / gridSize.x;
	pos.z = (hash - pos.x - (pos.y * gridSize.x)) / areaXY;

	return pos;
}

/* Loads the value of type T from char* (unoptimized) */
template<typename T>
__device__ T load(char* d_begin) {
	const uint size = sizeof(T);
	char values[size];
	for (uint i = 0; i < size; i++) {
		values[i] = d_begin[i];
	}
	return *(T*)values;
}

/* Saves the value of type T to char* (unoptimized) */
template<typename T>
__device__ void save(T value, char* d_begin) {
	const uint size = sizeof(T);
	char* values;
	values = (char*)&value;
	for (uint i = 0; i < size; i++) {
		d_begin[i] = values[i];
	}
}

/* Calculates for a given particle the d_HashList value and sets its id in d_IdList */
__global__ void fillHashGrid(char* d_List, uint* d_HashList, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo, bool aligned) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pInfo.groupCount) {
		float3 f;
		if (aligned)
			f = *(float3*)(d_List + idx * pInfo.stride);
		else {
			f = load<float3>(d_List + idx * pInfo.stride);
		}


		int3 gridPos = calcGridPos(f, pSysInfo);
		d_HashList[idx] = calcGridHash(gridPos, pSysInfo);
		d_IdList[idx] = idx;
	}
}

/* Sets the cellBegin and cellEnd pointers with current and previous hash value */
__global__ void setCellPointers(uint* d_HashList, uint* d_CellBegin, uint* d_CellEnd, ParticleInfo pInfo) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint hash = d_HashList[idx];
	if (idx < pInfo.groupCount) {

		if (idx == 0)
		{
			d_CellBegin[hash] = 0;
			return;

		}
		uint prevHash = d_HashList[idx - 1];
		if (hash != prevHash) {
			d_CellEnd[prevHash] = idx;
			d_CellBegin[hash] = idx;
		}
		if (idx == pInfo.groupCount - 1) {
			d_CellEnd[hash] = idx + 1;
		}
	}
}

/* Calculates the number of neighbours in a given area */
__global__ void getNumberOfNeighbours(char* d_List, uint* d_CellBegin, uint* d_CellEnd, uint* d_IdList, uint* d_OutCount, ParticleInfo pInfo, PSystemInfo pSysInfo, int3 queryGridSize, Particle queryParticle, int3 lowerBox, bool aligned) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint blockRet;
	uint size = queryGridSize.x * queryGridSize.y * queryGridSize.z;
	if (threadIdx.x == 0)
		blockRet = 0;
	__syncthreads();

	int3 pos = lowerBox + calcGridPosFromHash(idx, queryGridSize);
	if (idx < size && pos.x >= 0 && pos.x < pSysInfo.gridSize.x && pos.y >= 0 && pos.y < pSysInfo.gridSize.y && pos.z >= 0 && pos.z < pSysInfo.gridSize.z) {

		uint hash = calcGridHash(pos, pSysInfo);
		uint n = 0;
		uint begin = d_CellBegin[hash];
		uint end = d_CellEnd[hash];
		Particle particle;
		particle.radius = pInfo.groupRadius;
		uint id;
		for (uint i = begin; i < end; i++) {
			id = d_IdList[i];
			if (aligned)
				particle.pos = *(float3*)(d_List + id * pInfo.stride);
			else
				particle.pos = load<float3>(d_List + id * pInfo.stride);
			if (particle.radius == -1)
				particle.radius = load<float>(d_List + 12 + id * pInfo.stride);

			if (euclideanDistance(particle.pos - queryParticle.pos) < particle.radius + queryParticle.radius)
				n++;

		}

		atomicAdd((uint*)&blockRet, n);

	}
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(d_OutCount, blockRet);
	}
}