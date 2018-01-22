#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
#include <random>

//#include "HashGridProcessor.h"

// thrust for sorting
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
//#include <cuda.h>

#include "Loader.h"
#include "CudaOperations.cuh"
#define N 1
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


// calculate position grid
__host__ __device__ int3 calcGridPos(float3 p, PSystemInfo pSysInfo)
{
	int3 gridPos;
	gridPos.x = floor((p.x - pSysInfo.worldOrigin.x) / pSysInfo.cellSize.x);
	gridPos.y = floor((p.y - pSysInfo.worldOrigin.y) / pSysInfo.cellSize.y);
	gridPos.z = floor((p.z - pSysInfo.worldOrigin.z) / pSysInfo.cellSize.z);
	return gridPos;
}

// calculate hash value in grid
__host__ __device__ uint calcGridHash(int3 gridPos, PSystemInfo pSysInfo)
{
	return gridPos.z * pSysInfo.gridSize.y * pSysInfo.gridSize.x + gridPos.y * pSysInfo.gridSize.x + gridPos.x;
}

// calculate hash value in grid
__host__ __device__ int3 calcGridPosFromHash(uint hash, int3 gridSize)
{
	int3 pos;
	uint areaXY = gridSize.x * gridSize.y;
	pos.x = hash % gridSize.x;
	pos.y = ((hash - pos.x) % (areaXY)) / gridSize.x;
	pos.z = (hash - pos.x - (pos.y * gridSize.x)) / areaXY;

	return pos;
}

// loads the value of type T from char*
template<typename T>
__device__ T load(char* d_begin) {
	const uint size = sizeof(T);
	char values[size];
	for (uint i = 0; i < size; i++) {
		values[i] = d_begin[i];
	}
	return *(T*)values;
}

// saves the value of type T to char*
template<typename T>
__device__ void save(T value, char* d_begin) {
	const uint size = sizeof(T);
	char* values;
	values = (char*)&value;
	for (uint i = 0; i < size; i++) {
		d_begin[i] = values[i];
	}
}

__device__ bool checkCollision(float3 p1, float3 size1, float3 p2, float3 size2) {
	if (abs(p1.x - p2.x) < size1.x + size2.x) {
		if (abs(p1.y - p2.y) < size1.y + size2.y) {
			if (abs(p1.z - p2.z) < size1.z + size2.z) {
				return true;
			}
		}
	}
	return false;
}

__global__ void fillHashGrid(char* d_List, uint* d_HashList, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo, bool aligned) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pInfo.groupCount) {
		float3 f;
		if(aligned)
			f = *(float3*)(d_List + idx * pInfo.stride);
		else {
			f = load<float3>(d_List + idx * pInfo.stride);
		}
			
		
		int3 gridPos = calcGridPos(f, pSysInfo);
		d_HashList[idx] = calcGridHash(gridPos, pSysInfo);
		d_IdList[idx] = idx;
	}
}

__global__ void setCellPointers(uint* d_HashList, uint* d_CellBegin, uint* d_CellEnd, ParticleInfo pInfo) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void getNumberOfNeighboursOld(char* d_List, uint* d_CellBegin, uint* d_CellEnd, uint* d_IdList, uint* d_Hash, uint* d_OutCount, ParticleInfo pInfo, uint size, Particle queryParticle, bool aligned) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint blockRet;

	if (threadIdx.x == 0)
		blockRet = 0;
	__syncthreads();

	if (idx < size) {
		
		uint hash = d_Hash[idx];
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
			if(particle.radius == -1)
				particle.radius = load<float>(d_List + 12 + id * pInfo.stride);

			if (euclideanDistance(particle.pos - queryParticle.pos) < particle.radius + queryParticle.radius)
				n++;

		}
		
//		printf("Hello thread %d\n", threadIdx.x);
		atomicAdd((uint*)&blockRet, n);


//		*d_OutCount = n;

		__syncthreads();
		if (threadIdx.x == 0) {
//			uint ret;
//			for (int i = 0; i < BLOCKSIZE; i++) {
//				ret += threadRet[i];
//			}
			atomicAdd(d_OutCount, blockRet);
		}
			
//		*d_OutCount = 5;

	}
}

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

		//		printf("Hello thread %d\n", threadIdx.x);
		atomicAdd((uint*)&blockRet, n);

	}
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(d_OutCount, blockRet);
	}
}

void writeToFile(const std::string& name, uint* cellBegin, uint* cellEnd, uint3 gridSize) {
	std::ofstream outputFile(name, std::ifstream::binary);
	for (int i = 0; i < gridSize.x * gridSize.y * gridSize.z; i++) {
		uint64_t num = cellEnd[i] - cellBegin[i];
		outputFile.write((char*) &num, 4);
	}
	outputFile.close();
}

void writeToFile(const std::string& name, std::vector<std::vector<uint>> grid, uint3 gridSize) {
	std::ofstream outputFile(name, std::ifstream::binary);
	for (int i = 0; i < gridSize.x * gridSize.y * gridSize.z; i++) {
		uint64_t num = grid[i].size();
		outputFile.write((char*)&num, 4);
	}
	outputFile.close();
}

ParticleList reduceParticles(const ParticleList pList, uint targetCount) {

	std::random_device rd;

	/* Random number generator */
	std::default_random_engine generator(rd());

	/* Distribution on which to apply the generator */
	std::uniform_int_distribution<unsigned int> distribution(0, pList.info.groupCount - 1);

	bool* b = new bool[pList.info.groupCount];
	for (int i = 0; i < pList.info.groupCount; i++) {
		b[i] = false;
	}

	for (int i = 0; i < pList.info.groupCount - targetCount; i++) {
		uint rNum = distribution(generator);
		while (b[rNum] == true) {
			rNum = distribution(generator);
		}
		b[rNum] = true;
	}
	char* output = new char[targetCount * pList.info.stride];
	int index = 0;
	for (int i = 0; i < pList.info.groupCount; i++) {
		if (b[i] == false) {
			for (int a = 0; a < pList.info.stride; a++) {
				output[index * pList.info.stride + a] = pList.data[i * pList.info.stride + a];
			}
			index++;
		}
	}
	ParticleInfo pInfo = pList.info;
	pInfo.groupCount = targetCount;
	ParticleList particles;
	particles.info = pInfo;
	particles.data = output;
	delete[] b;
	return particles;
}

ParticleList reduceParticles(const ParticleList pList, float redPercentage) {
	uint targetCount = (uint)(pList.info.groupCount * (1 - redPercentage));
	return reduceParticles(pList, targetCount);
}

uint getNumberOfNeighboursGPUOld(char* d_List, uint* d_CellBegin, uint* d_CellEnd, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo, Particle queryParticle, bool aligned) {
	long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	int3 queryParticlePos = calcGridPos(queryParticle.pos, pSysInfo);
	int3 queryGridSize = make_int3(2 * queryParticle.radius / pSysInfo.cellSize.x + 3, 2 * queryParticle.radius / pSysInfo.cellSize.y + 3, 2 * queryParticle.radius / pSysInfo.cellSize.z + 3);
	int3 queryLowerBB = make_int3(queryParticlePos.x - (int)(queryGridSize.x * 0.5), queryParticlePos.y - (int)(queryGridSize.y * 0.5), queryParticlePos.z - (int)(queryGridSize.z * 0.5));
	uint threadSize = queryGridSize.x * queryGridSize.y * queryGridSize.z;

	uint* d_OutCount;
	uint* d_Hash;
	uint* h_Hash = new uint[threadSize];
	uint h_OutCount = 0;

	uint a = 0;
	for (int z = queryLowerBB.z; z < queryLowerBB.z + queryGridSize.z; z++) {
		for (int y = queryLowerBB.y; y < queryLowerBB.y + queryGridSize.y; y++) {
			for (int x = queryLowerBB.x; x < queryLowerBB.x + queryGridSize.x; x++) {
				uint i = calcGridHash(make_int3(x, y, z), pSysInfo);
				h_Hash[a++] = i;
				/*
				std::cout << "0pos: " << x << ", " << y << ", " << z << std::endl;
				int3 pos = calcGridPosFromHash(i, pSysInfo);
				std::cout << "1pos: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
				*/
			}
		}
	}
	HANDLE_ERROR(cudaMalloc(&d_Hash, sizeof(uint) * threadSize));
	HANDLE_ERROR(cudaMemcpy(d_Hash, h_Hash, sizeof(uint) * threadSize, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc(&d_OutCount, sizeof(uint)));
	HANDLE_ERROR(cudaMemset(d_OutCount, 0, sizeof(uint)));

	// kernel call
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(threadSize / (float)BLOCKSIZE));
	bool isAligned = true;
	if (pInfo.stride % 2 != 0)
		isAligned = false;


	getNumberOfNeighboursOld << <dimGrid, dimBlock >> > (d_List, d_CellBegin, d_CellEnd, d_IdList, d_Hash, d_OutCount, pInfo, threadSize, queryParticle, isAligned);
//	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMemcpy(&h_OutCount, d_OutCount, sizeof(uint), cudaMemcpyDeviceToHost));

	long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::cout << "GPU Old: " << (endTime - startTime) / 1000.0f << std::endl;
	HANDLE_ERROR(cudaFree(d_Hash));
	HANDLE_ERROR(cudaFree(d_OutCount));
	delete[] h_Hash;
	return h_OutCount;
}

uint getNumberOfNeighboursGPU(char* d_List, uint* d_CellBegin, uint* d_CellEnd, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo, Particle queryParticle, bool aligned) {
	long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	int3 queryParticlePos = calcGridPos(queryParticle.pos, pSysInfo);
	int3 queryGridSize = make_int3(2 * queryParticle.radius / pSysInfo.cellSize.x + 3, 2 * queryParticle.radius / pSysInfo.cellSize.y + 3, 2 * queryParticle.radius / pSysInfo.cellSize.z + 3);
	int3 queryLowerBB = make_int3(queryParticlePos.x - (int) (queryGridSize.x * 0.5), queryParticlePos.y - (int)(queryGridSize.y * 0.5), queryParticlePos.z - (int)(queryGridSize.z * 0.5));
//	int3 queryGridSize = make_int3(pSysInfo.gridSize.x, pSysInfo.gridSize.y, pSysInfo.gridSize.z);
//	int3 queryLowerBB = make_int3(0, 0, 0);
	uint threadSize = queryGridSize.x * queryGridSize.y * queryGridSize.z;

	uint* d_OutCount;
	uint h_OutCount = 0;

	HANDLE_ERROR(cudaMalloc(&d_OutCount, sizeof(uint)));
	HANDLE_ERROR(cudaMemset(d_OutCount, 0, sizeof(uint)));

	// kernel call
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(threadSize / (float)BLOCKSIZE));
	bool isAligned = true;
	if (pInfo.stride % 2 != 0)
		isAligned = false;


	getNumberOfNeighbours << <dimGrid, dimBlock >> > (d_List, d_CellBegin, d_CellEnd, d_IdList, d_OutCount, pInfo, pSysInfo, queryGridSize, queryParticle, queryLowerBB, isAligned);
	//	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMemcpy(&h_OutCount, d_OutCount, sizeof(uint), cudaMemcpyDeviceToHost));

	long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::cout << "GPU: " << (endTime - startTime) / 1000.0f << std::endl;
	HANDLE_ERROR(cudaFree(d_OutCount));
	return h_OutCount;
}

uint getNumberOfNeighboursCPU(char* h_List, uint* d_CellBegin, uint* d_CellEnd, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo, Particle queryParticle) {
	uint n = 0;
	uint hashSize = pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z;

	uint* h_CellBegin = new uint[hashSize];
	uint* h_CellEnd = new uint[hashSize];
	uint* h_IdList = new uint[pInfo.groupCount];
	HANDLE_ERROR(cudaMemcpy(h_CellBegin, d_CellBegin, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_CellEnd, d_CellEnd, sizeof(uint) * hashSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_IdList, d_IdList, sizeof(uint) * pInfo.groupCount, cudaMemcpyDeviceToHost));

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
						float3 pos = *(float3*)(h_List + h_IdList[a] * pInfo.stride);
						float radius = pInfo.groupRadius;
						if (radius == -1)
							radius = *(float*)(h_List + h_IdList[a] * pInfo.stride + 12);
						if (euclideanDistance(pos - queryParticle.pos) < radius + queryParticle.radius)
							n++;
					}
				}
			}
		}
	}
	long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::cout << "CPU: " << (endTime - startTime) / 1000.0f << std::endl;

	delete[] h_CellBegin;
	delete[] h_CellEnd;
	delete[] h_IdList;
	return n;
}



long long benchmarkPListGPU(ParticleList p, PSystemInfo pSysInfo, int iterations = 1) {
	unsigned long long minTime = -1;
	for (int i = 0; i < iterations; i++) {
		char* d_List;

		uint* d_HashList;
		uint* d_IdList;
		uint* d_CellBegin;
		uint* d_CellEnd;

		
		HANDLE_ERROR(cudaMalloc(&d_List, p.info.stride * p.info.groupCount));
		HANDLE_ERROR(cudaMemcpy(d_List, p.data, p.info.stride * p.info.groupCount, cudaMemcpyHostToDevice));

		
		HANDLE_ERROR(cudaMalloc(&d_HashList, sizeof(uint) * p.info.groupCount * N));

		HANDLE_ERROR(cudaMalloc(&d_IdList, sizeof(uint) * p.info.groupCount * N));

		HANDLE_ERROR(cudaMalloc(&d_CellBegin, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMemset(d_CellBegin, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMalloc(&d_CellEnd, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMemset(d_CellEnd, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));


		// kernel call
		dim3 dimBlock(BLOCKSIZE);
		dim3 dimGrid(ceil(p.info.groupCount / (float)BLOCKSIZE));
		bool isAligned = true;
		if (p.info.stride % 2 != 0)
			isAligned = false;

		
		fillHashGrid << <dimGrid, dimBlock >> > (d_List, d_HashList, d_IdList, p.info, pSysInfo, isAligned);


		thrust::sort_by_key(thrust::device_ptr<uint>(d_HashList),
			thrust::device_ptr<uint>(d_HashList + p.info.groupCount * N),
			thrust::device_ptr<uint>(d_IdList));
		
		setCellPointers << <dimGrid, dimBlock >> > (d_HashList, d_CellBegin, d_CellEnd, p.info);
		cudaDeviceSynchronize();

		Particle particle;
		particle.pos = make_float3(60, 600, 60);
		particle.radius = 60;
		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//		std::cout << getNumberOfNeighboursGPUOld(d_List, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle, isAligned) << std::endl;
		std::cout << getNumberOfNeighboursGPU(d_List, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle, isAligned) << std::endl;
		std::cout << getNumberOfNeighboursCPU(p.data, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle) << std::endl;

		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		if (endTime - startTime < minTime)
			minTime = endTime - startTime;

		HANDLE_ERROR(cudaFree(d_List));

		HANDLE_ERROR(cudaFree(d_HashList));
		HANDLE_ERROR(cudaFree(d_IdList));
		HANDLE_ERROR(cudaFree(d_CellBegin));
		HANDLE_ERROR(cudaFree(d_CellEnd));
	}
	std::cout << "GPU: Particle count: " << p.info.groupCount << " Time: " << minTime / 1000.0f << std::endl;
	return minTime;
}

long long benchmarkPListCPU(ParticleList p, PSystemInfo pSysInfo, int iterations = 1) {
	unsigned long long minTime = -1;
	for (int a = 0; a < iterations; a++) {

		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		std::vector<std::vector<uint>> grid(pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z);

		for (uint i = 0; i < p.info.groupCount; i++)
			grid[calcGridHash(calcGridPos(*(float3*)(p.data + i * p.info.stride), pSysInfo), pSysInfo)].push_back(i);

		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		if (endTime - startTime < minTime)
			minTime = endTime - startTime;

//		writeToFile("testFile.raw", grid, pSysInfo.gridSize);
	}
	std::cout << "CPU: Particle count: " << p.info.groupCount << " Time: " << minTime / 1000.0f << std::endl;
	return minTime;
}

int main(int argc, char **argv)
{
	

	Loader loader("exp2mill.mmpld");

	auto pLists = loader.getFrame(20);

	uint3 gridSize;
	gridSize.x = 32;
	gridSize.y = 320;
	gridSize.z = 32;
	PSystemInfo pSysInfo = loader.calcBSystemInfo(gridSize);
	
	/*
	for (int i = 1; i < 11; i++) {
		std::cout << "Grid Size: (" << pSysInfo.gridSize.x << ", " << pSysInfo.gridSize.y << ", " << pSysInfo.gridSize.z << ") Origin: (" << pSysInfo.worldOrigin.x << ", " << pSysInfo.worldOrigin.y << ", " << pSysInfo.worldOrigin.z << ") Cell Size: (" << pSysInfo.cellSize.x << ", " << pSysInfo.cellSize.y << ", " << pSysInfo.cellSize.z << ")" << std::endl;
		benchmarkPListGPU(pLists[0], pSysInfo, 128, 100);
		benchmarkPListCPU(pLists[0], pSysInfo, 100);
		gridSize.x *= 2;
		gridSize.y *= 2;
		gridSize.z *= 2;
		pSysInfo = loader.calcBSystemInfo(gridSize);
	}
	*/
	benchmarkPListGPU(pLists[0], pSysInfo, 100);
	
	/*
	std::ofstream outputFile("benchmark.csv");
	outputFile << "Partikel Anzahl; GPU\n";

	
	for (uint i = 1; i < 10; i++) {
		ParticleList pList = reduceParticles(pLists[0], 39870 + i);
		outputFile << pList.info.groupCount << ";";
		outputFile << benchmarkPListGPU(pList, pSysInfo, 128, 100) << "\n";
//		outputFile << benchmarkPListCPU(pList, pSysInfo, 100) << "\n";
	}
	outputFile.close();
	*/
	return 0;
}
