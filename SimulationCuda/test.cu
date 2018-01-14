#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <random>

//#include "HashGridProcessor.h"

// thrust for sorting
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
//#include <cuda.h>

#include "Loader.h"
#define N 1

#define HANDLE_ERROR(err)\
	(handleCudaError(err, __FILE__, __LINE__))


static void handleCudaError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

__host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
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

__device__ float3 calcGridPosFloat(float3 p, PSystemInfo pSysInfo)
{
	float3 gridPos;
	gridPos.x = (p.x - pSysInfo.worldOrigin.x) / pSysInfo.cellSize.x;
	gridPos.y = (p.y - pSysInfo.worldOrigin.y) / pSysInfo.cellSize.y;
	gridPos.z = (p.z - pSysInfo.worldOrigin.z) / pSysInfo.cellSize.z;
	return gridPos;
}

// calculate hash value in grid
__host__ __device__ uint calcGridHash(int3 gridPos, PSystemInfo pSysInfo)
{
	return gridPos.z * pSysInfo.gridSize.y * pSysInfo.gridSize.x + gridPos.y * pSysInfo.gridSize.x + gridPos.x;
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

ParticleList reduceParticles(const ParticleList pList, float redPercentage) {
	
	std::random_device rd;

	/* Random number generator */
	std::default_random_engine generator(rd());

	/* Distribution on which to apply the generator */
	std::uniform_int_distribution<unsigned int> distribution(0, pList.info.groupCount - 1);

	bool* b = new bool[pList.info.groupCount];
	for (int i = 0; i < pList.info.groupCount; i++) {
		b[i] = false;
	}

	int numRemove = (int)(pList.info.groupCount * redPercentage);
	for (int i = 0; i < numRemove; i++) {
		uint rNum = distribution(generator);
		while (b[rNum] == true) {
			rNum = distribution(generator);
		}
		b[rNum] = true;
	}
	char* output = new char[(pList.info.groupCount - numRemove) * pList.info.stride];
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
	pInfo.groupCount = pList.info.groupCount - numRemove;
	ParticleList particles;
	particles.info = pInfo;
	particles.data = output;
	delete[] b;
	return particles;
}

void benchmarkTimeGPU() {
	int blocksize = 128;
	unsigned long long minTime = -1;
	for (int a = 0; a < 100; a++) {
/*	while(true){
		uint number = 0;
		std::cin >> number;*/
		
		Loader loader("exp2mill.mmpld");

		auto pList = loader.getFrame(20);

		/*
		float3 cellSize;
		cellSize.x = 1;
		cellSize.y = 1;
		cellSize.z = 1;
		PSystemInfo pSysInfo = loader.calcBSystemInfo(cellSize);
		*/
		
		uint3 gridSize;
		gridSize.x = 32;
		gridSize.y = 320;
		gridSize.z = 32;
		PSystemInfo pSysInfo = loader.calcBSystemInfo(gridSize);
		
//		PSystemInfo pSysInfo = loader.calcBSystemInfo(pList);
		

		std::cout << "Grid Size: (" << pSysInfo.gridSize.x << ", " << pSysInfo.gridSize.y << ", " << pSysInfo.gridSize.z << ") Origin: (" << pSysInfo.worldOrigin.x << ", " << pSysInfo.worldOrigin.y << ", " << pSysInfo.worldOrigin.z << ") Cell Size: (" << pSysInfo.cellSize.x << ", " << pSysInfo.cellSize.y << ", " << pSysInfo.cellSize.z << ")" << std::endl;
		
		std::vector<char*> d_List(pList.size());
		std::vector<char*> h_List(pList.size());

		std::vector<uint*> d_HashList(pList.size());
		std::vector<uint*> d_IdList(pList.size());
		std::vector<uint*> d_CellBegin(pList.size());
		std::vector<uint*> d_CellEnd(pList.size());
		
		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		for (int n = 0; n < pList.size(); n++) {
			auto p = pList[n];
			
			HANDLE_ERROR(cudaMalloc(&d_List[n], p.info.stride * p.info.groupCount));
			HANDLE_ERROR(cudaMemcpy(d_List[n], p.data, p.info.stride * p.info.groupCount, cudaMemcpyHostToDevice));

			
			HANDLE_ERROR(cudaMalloc(&d_HashList[n], sizeof(uint) * p.info.groupCount * N));
//			HANDLE_ERROR(cudaMemset(d_HashList[n], -1, sizeof(uint) * p.info.groupCount * N));
			HANDLE_ERROR(cudaMalloc(&d_IdList[n], sizeof(uint) * p.info.groupCount * N));
//			HANDLE_ERROR(cudaMemset(d_IdList[n], -1, sizeof(uint) * p.info.groupCount * N));
			HANDLE_ERROR(cudaMalloc(&d_CellBegin[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
			HANDLE_ERROR(cudaMemset(d_CellBegin[n], -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
			HANDLE_ERROR(cudaMalloc(&d_CellEnd[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
			HANDLE_ERROR(cudaMemset(d_CellEnd[n], -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));

			
			// kernel call
			dim3 dimBlock(blocksize);
			dim3 dimGrid(ceil(pList[n].info.groupCount / (float)blocksize));
			bool isAligned = true;
			if (p.info.stride % 2 != 0)
				isAligned = false;

			
			fillHashGrid << <dimGrid, dimBlock >> > (d_List[n], d_HashList[n], d_IdList[n], p.info, pSysInfo, isAligned);

//			cudaDeviceSynchronize();
			
			thrust::sort_by_key(thrust::device_ptr<uint>(d_HashList[n]),
				thrust::device_ptr<uint>(d_HashList[n] + p.info.groupCount * N),
				thrust::device_ptr<uint>(d_IdList[n]));
			
			setCellPointers << <dimGrid, dimBlock >> > (d_HashList[n], d_CellBegin[n], d_CellEnd[n], p.info);
			cudaDeviceSynchronize();

//			cudaDeviceSynchronize();
			/*
			uint* h_HashList = new uint[p.info.groupCount * N];
			uint* h_IdList = new uint[p.info.groupCount * N];
			HANDLE_ERROR(cudaMemcpy(h_HashList, d_HashList[n], sizeof(uint) * p.info.groupCount * N, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_IdList, d_IdList[n], sizeof(uint) * p.info.groupCount * N, cudaMemcpyDeviceToHost));

			for (int i = 0; i < p.info.groupCount * N; i++) {
				std::cout << i << ": Hash: " << h_HashList[i] << ", Id: " << h_IdList[i] << std::endl;
			}

			delete[] h_HashList;
			delete[] h_IdList;
			*/
			/*
			uint* h_CellBegin = new uint[pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z];
			uint* h_CellEnd = new uint[pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z];
			HANDLE_ERROR(cudaMemcpy(h_CellBegin, d_CellBegin[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_CellEnd, d_CellEnd[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z, cudaMemcpyDeviceToHost));
			writeToFile("test.raw", h_CellBegin, h_CellEnd, pSysInfo.gridSize);
			
			uint maxParticle = 0;
			uint maxI = 0;

			for (int i = 0; i < pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z; i++) {
//				std::cout << "Hash: " << i << ", Begin: " << h_CellBegin[i] << ", End: " << h_CellEnd[i] << ", Particles: " << h_CellEnd[i] - h_CellBegin[i] << std::endl;
				if (maxParticle < h_CellEnd[i] - h_CellBegin[i]) {
					maxParticle = h_CellEnd[i] - h_CellBegin[i];
					maxI = i;
				}
					
			}
			std::cout << "Max particle per cube: " << maxParticle << " at " << maxI << std::endl;
			delete[] h_CellBegin;
			delete[] h_CellEnd;
			*/

			/*
			// copy array back for verification
			h_List[n] = new char[p.info.stride * p.info.groupCount];
			HANDLE_ERROR(cudaMemcpy(h_List[n], d_List[n], p.info.stride * p.info.groupCount, cudaMemcpyDeviceToHost));
			
			// value testing
			for (int i = 0; i < p.info.groupCount; i++) {
				float3* pos = (float3*)(p.data + i * p.info.stride);
				float3* posNew = (float3*)(h_List[n] + i * p.info.stride);
				if (pos->x != posNew->x || pos->y != posNew->y || pos->z != posNew->z)
					std::cout << i << ": " << pos->x << ", " << pos->y << ", " << pos->z << "; " << posNew->x << ", " << posNew->y << ", " << posNew->z << std::endl;
			}
			*/
			
		
		
			std::cout << "New list with " << p.info.groupCount << " particles" << std::endl;
		}
		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		std::cout << (endTime - startTime) / 1000.0 << std::endl;

		if (endTime - startTime < minTime)
			minTime = endTime - startTime;
		
		for (int n = 0; n < pList.size(); n++) {
			HANDLE_ERROR(cudaFree(d_List[n]));

			HANDLE_ERROR(cudaFree(d_HashList[n]));
			HANDLE_ERROR(cudaFree(d_IdList[n]));
			HANDLE_ERROR(cudaFree(d_CellBegin[n]));
			HANDLE_ERROR(cudaFree(d_CellEnd[n]));
			delete[] h_List[n];
			delete[] pList[n].data;
		}


	}

	std::cout << minTime / 1000.0f << std::endl;
}

long long benchmarkPListGPU(ParticleList p, PSystemInfo pSysInfo, int blocksize, int iterations = 1) {
	unsigned long long minTime = -1;
	for (int i = 0; i < iterations; i++) {
		char* d_List;

		uint* d_HashList;
		uint* d_IdList;
		uint* d_CellBegin;
		uint* d_CellEnd;

		
		long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		HANDLE_ERROR(cudaMalloc(&d_List, p.info.stride * p.info.groupCount));
		HANDLE_ERROR(cudaMemcpy(d_List, p.data, p.info.stride * p.info.groupCount, cudaMemcpyHostToDevice));

		
		HANDLE_ERROR(cudaMalloc(&d_HashList, sizeof(uint) * p.info.groupCount * N));

		HANDLE_ERROR(cudaMalloc(&d_IdList, sizeof(uint) * p.info.groupCount * N));

		HANDLE_ERROR(cudaMalloc(&d_CellBegin, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMemset(d_CellBegin, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMalloc(&d_CellEnd, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));
		HANDLE_ERROR(cudaMemset(d_CellEnd, -1, sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z));


		// kernel call
		dim3 dimBlock(blocksize);
		dim3 dimGrid(ceil(p.info.groupCount / (float)blocksize));
		bool isAligned = true;
		if (p.info.stride % 2 != 0)
			isAligned = false;

		
		fillHashGrid << <dimGrid, dimBlock >> > (d_List, d_HashList, d_IdList, p.info, pSysInfo, isAligned);


		thrust::sort_by_key(thrust::device_ptr<uint>(d_HashList),
			thrust::device_ptr<uint>(d_HashList + p.info.groupCount * N),
			thrust::device_ptr<uint>(d_IdList));

		setCellPointers << <dimGrid, dimBlock >> > (d_HashList, d_CellBegin, d_CellEnd, p.info);
		cudaDeviceSynchronize();

		long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//		std::cout << (endTime - startTime) / 1000.0 << std::endl;

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
	

	Loader loader("laser.mmpld");

	auto pLists = loader.getFrame(20);

	uint3 gridSize;
	gridSize.x = 32;
	gridSize.y = 32;
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
	std::ofstream outputFile("benchmark.csv");
	outputFile << "Partikel Anzahl; GPU; CPU\n";

	
	for (int i = 1; i < 100; i++) {
		ParticleList pList = reduceParticles(pLists[0], 1 - i * 0.001f);
		outputFile << pList.info.groupCount << ";";
		outputFile << benchmarkPListGPU(pList, pSysInfo, 256, 100) << ";";
		outputFile << benchmarkPListCPU(pList, pSysInfo, 100) << "\n";
	}
	outputFile.close();
	return 0;
}
