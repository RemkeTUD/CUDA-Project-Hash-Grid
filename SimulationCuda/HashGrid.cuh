#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
#include <random>

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "Kernel.cuh"



class HashGrid {
public:
	/* Create hashgrid on the GPU */
	HashGrid(ParticleList p, PSystemInfo pSysInfo);
	/* Free the memory on the GPU*/
	~HashGrid();

	/* Calculate and returns the number of neighbours for a given queary particle on the GPU*/
	uint getNumberOfNeighboursGPU(Particle queryParticle, float& time, uint iterations);
	/* Calculate and returns the number of neighbours for a given queary particle on the CPU*/
	uint getNumberOfNeighboursCPU(Particle queryParticle, float& time, uint iterations);
	/* Write hash grid volume data to raw file*/
	void writeToRawFile(const std::string& name);

	float initHashGridTime = 0;

private:
	ParticleList p;
	PSystemInfo pSysInfo;
	/* Cuda device pointer d_...*/
	char* d_List;
	uint* d_HashList;
	uint* d_IdList;
	uint* d_CellBegin;
	uint* d_CellEnd;
	/* Is dataset aligned */
	bool isAligned = true;
};


/* Implementation */
HashGrid::HashGrid(ParticleList p, PSystemInfo pSysInfo)
{
	HANDLE_ERROR(cudaMalloc(&d_List, p.info.stride * p.info.groupCount));
	HANDLE_ERROR(cudaMemcpy(d_List, p.data, p.info.stride * p.info.groupCount, cudaMemcpyHostToDevice));

	long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	HANDLE_ERROR(cudaMalloc(&d_HashList, sizeof(uint) * p.info.groupCount));

	HANDLE_ERROR(cudaMalloc(&d_IdList, sizeof(uint) * p.info.groupCount));

	HANDLE_ERROR(cudaMalloc(&d_CellBegin, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
	HANDLE_ERROR(cudaMemset(d_CellBegin, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
	HANDLE_ERROR(cudaMalloc(&d_CellEnd, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
	HANDLE_ERROR(cudaMemset(d_CellEnd, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));


	// kernel call
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(p.info.groupCount / (float)BLOCKSIZE));
	if (p.info.stride % 2 != 0)
		isAligned = false;


	fillHashGrid << <dimGrid, dimBlock >> > (d_List, d_HashList, d_IdList, p.info, pSysInfo, isAligned);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_HashList),
		thrust::device_ptr<uint>(d_HashList + p.info.groupCount),
		thrust::device_ptr<uint>(d_IdList));

	setCellPointers << <dimGrid, dimBlock >> > (d_HashList, d_CellBegin, d_CellEnd, p.info);
	HANDLE_ERROR(cudaDeviceSynchronize());
	long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	initHashGridTime = (endTime - startTime) / 1000.0f;
	this->p = p;
	this->pSysInfo = pSysInfo;
}

HashGrid::~HashGrid()
{
	HANDLE_ERROR(cudaFree(d_List));

	HANDLE_ERROR(cudaFree(d_HashList));
	HANDLE_ERROR(cudaFree(d_IdList));
	HANDLE_ERROR(cudaFree(d_CellBegin));
	HANDLE_ERROR(cudaFree(d_CellEnd));
}

uint HashGrid::getNumberOfNeighboursGPU(Particle queryParticle, float& time, uint iterations)
{
	time = std::numeric_limits<float>::max();
	uint h_OutCount;
	for (int i = 0; i < iterations; i++) {
		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		int3 queryParticlePos = calcGridPos(queryParticle.pos, pSysInfo);
		int3 queryGridSize = make_int3(2 * queryParticle.radius / pSysInfo.cellSize.x + 3, 2 * queryParticle.radius / pSysInfo.cellSize.y + 3, 2 * queryParticle.radius / pSysInfo.cellSize.z + 3);
		int3 queryLowerBB = make_int3(queryParticlePos.x - (int)(queryGridSize.x * 0.5), queryParticlePos.y - (int)(queryGridSize.y * 0.5), queryParticlePos.z - (int)(queryGridSize.z * 0.5));
		//	int3 queryGridSize = make_int3(pSysInfo.gridSize.x, pSysInfo.gridSize.y, pSysInfo.gridSize.z);
		//	int3 queryLowerBB = make_int3(0, 0, 0);
		uint threadSize = queryGridSize.x * queryGridSize.y * queryGridSize.z;

		uint* d_OutCount;
		h_OutCount = 0;

		HANDLE_ERROR(cudaMalloc(&d_OutCount, sizeof(uint)));
		HANDLE_ERROR(cudaMemset(d_OutCount, 0, sizeof(uint)));

		// kernel call
		dim3 dimBlock(BLOCKSIZE);
		dim3 dimGrid(ceil(threadSize / (float)BLOCKSIZE));
		bool isAligned = true;
		if (p.info.stride % 2 != 0)
			isAligned = false;


		getNumberOfNeighbours << <dimGrid, dimBlock >> > (d_List, d_CellBegin, d_CellEnd, d_IdList, d_OutCount, p.info, pSysInfo, queryGridSize, queryParticle, queryLowerBB, isAligned);
		//	cudaDeviceSynchronize();
		HANDLE_ERROR(cudaMemcpy(&h_OutCount, d_OutCount, sizeof(uint), cudaMemcpyDeviceToHost));

		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//		std::cout << "GPU: " << (endTime - startTime) / 1000.0f << std::endl;
		HANDLE_ERROR(cudaFree(d_OutCount));
		if ((endTime - startTime) / 1000.0f < time)
			time = (endTime - startTime) / 1000.0f;
	}



	return h_OutCount;
}

uint HashGrid::getNumberOfNeighboursCPU(Particle queryParticle, float& time, uint iterations)
{
	time = std::numeric_limits<float>::max();
	uint n = 0;
	for (int i = 0; i < iterations; i++) {
		n = 0;
		uint hashSize = pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z;

		uint* h_CellBegin = new uint[hashSize];
		uint* h_CellEnd = new uint[hashSize];
		uint* h_IdList = new uint[p.info.groupCount];
		HANDLE_ERROR(cudaMemcpy(h_CellBegin, d_CellBegin, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_CellEnd, d_CellEnd, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_IdList, d_IdList, sizeof(uint) * p.info.groupCount, cudaMemcpyDeviceToHost));

		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		int3 queryParticlePos = calcGridPos(queryParticle.pos, pSysInfo);
		int3 queryGridSize = make_int3(2 * queryParticle.radius / pSysInfo.cellSize.x + 3, 2 * queryParticle.radius / pSysInfo.cellSize.y + 3, 2 * queryParticle.radius / pSysInfo.cellSize.z + 3);
		int3 queryLowerBB = make_int3(queryParticlePos.x - (int)(queryGridSize.x * 0.5), queryParticlePos.y - (int)(queryGridSize.y * 0.5), queryParticlePos.z - (int)(queryGridSize.z * 0.5));
		for (int z = queryLowerBB.z; z < queryLowerBB.z + queryGridSize.z; z++) {
			for (int y = queryLowerBB.y; y < queryLowerBB.y + queryGridSize.y; y++) {
				for (int x = queryLowerBB.x; x < queryLowerBB.x + queryGridSize.x; x++) {
					if (x >= 0 && x < pSysInfo.gridSize.x && y >= 0 && y < pSysInfo.gridSize.y && z >= 0 && z < pSysInfo.gridSize.z) {
						uint i = calcGridHash(make_int3(x, y, z), pSysInfo);
						uint begin = h_CellBegin[i];
						uint end = h_CellEnd[i];
						for (uint a = begin; a < end; a++) {
							float3 pos = *(float3*)(p.data + h_IdList[a] * p.info.stride);
							float radius = p.info.groupRadius;
							if (radius == -1)
								radius = *(float*)(p.data + h_IdList[a] * p.info.stride + 12);
							if (euclideanDistance(pos - queryParticle.pos) < radius + queryParticle.radius)
								n++;
						}
					}
				}
			}
		}
		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//		std::cout << "CPU: " << (endTime - startTime) / 1000.0f << std::endl;

		delete[] h_CellBegin;
		delete[] h_CellEnd;
		delete[] h_IdList;
		if ((endTime - startTime) / 1000.0f < time)
			time = (endTime - startTime) / 1000.0f;
	}
	return n;
}

void HashGrid::writeToRawFile(const std::string & name)
{
	uint hashSize = pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z;

	uint* h_CellBegin = new uint[hashSize];
	uint* h_CellEnd = new uint[hashSize];

	HANDLE_ERROR(cudaMemcpy(h_CellBegin, d_CellBegin, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_CellEnd, d_CellEnd, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));

	std::ofstream outputFile(name, std::ifstream::binary);
	for (int i = 0; i < hashSize; i++) {
		uint64_t num = h_CellEnd[i] - h_CellBegin[i];
		outputFile.write((char*)&num, 4);
	}
	outputFile.close();
	delete[] h_CellBegin;
	delete[] h_CellEnd;
}