#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include "cuda_runtime.h"

typedef unsigned int uint;

struct ParticleInfo {
	uint32_t stride = 0;
	float groupRadius = -1;
	uint64_t groupCount = 0;
};

struct PSystemInfo {
	uint3 gridSize;
	float3 worldOrigin;
	float3 cellSize;
};

struct ParticleList {
	ParticleInfo info;
	char* data;
};

class Loader {
private:
	uint32_t m_frameCount;
	float3 m_lowBBox, m_highBBox;
	float3 m_lowCBox, m_highCBox;
	std::ifstream m_file;
	std::vector<uint64_t> m_frameTable;
	
public:
	Loader(const char* filename);
	~Loader();

	// returns particle lists for wanted frame
	std::vector<ParticleList> getFrame(uint32_t frameId);


	bool Loader::isInBBox(std::vector<ParticleList> pList, PSystemInfo pSysInfo);

	// calculates particle system info with bounding box and wanted gridSize; returns it
	PSystemInfo calcBSystemInfo(uint3 gridSize);
	// calculates particle system info with bounding box and wanted cellSize; returns it
	PSystemInfo calcBSystemInfo(float3 cellSize);

	// calculates particle system info with minimal cellSize for best performance (can take longer to create)
	PSystemInfo calcBSystemInfo(std::vector<ParticleList> pList);


};