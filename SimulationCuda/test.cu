#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// thrust for sorting
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
//#include <cuda.h>

#include "Loader.h"
#define N 1

typedef unsigned int uint;

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
__device__ int3 calcGridPos(float3 p, PSystemInfo pSysInfo)
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
__device__ uint calcGridHash(int3 gridPos, PSystemInfo pSysInfo)
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

__global__ void inc(char* d_List, ParticleInfo pInfo, PSystemInfo pSysInfo) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pInfo.groupCount) {
		float3 f = load<float3>(d_List + idx * pInfo.stride);
		int3 i = calcGridPos(f, pSysInfo);
		f.x = i.x;
		f.y = i.y;
		f.z = i.z;
		save<float3>(f, d_List + idx * pInfo.stride);
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

__global__ void fillHashGrid(char* d_List, uint* d_HashList, uint* d_IdList, ParticleInfo pInfo, PSystemInfo pSysInfo) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pInfo.groupCount) {
		float3 f = load<float3>(d_List + idx * pInfo.stride);
		int3 gridPos = calcGridPos(f, pSysInfo);
		d_HashList[idx] = calcGridHash(gridPos, pSysInfo);
		d_IdList[idx] = idx;

		/*
		float radius = pInfo.groupRadius;
		if (radius == -1)
			radius = load<float>(d_List + idx * pInfo.stride + 12);

		float3 rad;
		rad.x = radius / pSysInfo.cellSize.x;
		rad.y = radius / pSysInfo.cellSize.y;
		rad.z = radius / pSysInfo.cellSize.z;
		int counter = 0;
		float3 floatPos = calcGridPosFloat(f, pSysInfo);
//		if (idx == 5)
//			printf("Hello from idx %d, with ownPos %f, %f, %f\n", idx, floatPos.x, floatPos.y, floatPos.z);
//		if (idx == 5)
//			printf("Hello from idx %d, with radius %f, %f, %f\n", idx, rad.x, rad.y, rad.z);
		for (int x = -1; x < 2; x++) {
			for (int y = -1; y < 2; y++) {
				for (int z = -1; z < 2; z++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					float3 neighbourPosf = make_float3(neighbourPos.x + 0.5f, neighbourPos.y + 0.5f, neighbourPos.z + 0.5f);

					if (checkCollision(floatPos, rad, neighbourPosf, make_float3(0.5f,0.5f,0.5f))) {
//								if (blockIdx.x == 2 && threadIdx.x == 61)
//						printf("Hello from idx %d, with count %d\n", idx, counter);
						d_HashList[pInfo.groupCount * counter + idx] = calcGridHash(neighbourPos, pSysInfo);
						d_IdList[pInfo.groupCount * counter + idx] = idx;
						counter++;
//						if (idx == 5)
//							printf("Hello from idx %d, with neighbourPos %f, %f, %f\n", idx, neighbourPosf.x, neighbourPosf.y, neighbourPosf.z);
					}
				}
			}
			
		}
//		if (counter > 8)
//			printf("Hello from idx %d, with count %d\n", idx, counter);
		
		*/

	}
}

__global__ void setCellPointersOld(uint* d_HashList, uint* d_CellBegin, uint* d_CellEnd, ParticleInfo pInfo) {
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

__global__ void setCellPointers(uint* d_HashList, uint* d_CellBegin, uint* d_CellEnd, ParticleInfo pInfo) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint hash = d_HashList[idx];
	if (idx < pInfo.groupCount * N - 1 && hash != -1) {
		
		if (idx == 0)
		{
			d_CellBegin[hash] = 0;
			return;
		}
		else{
			uint nextHash = d_HashList[idx + 1];
			if (hash != nextHash) {
				d_CellEnd[hash] = idx + 1;
				if(nextHash != -1)
					d_CellBegin[nextHash] = idx + 1;
			}
		}
		// not optimal last element is never in a list
		if (idx == pInfo.groupCount * N - 1){
			d_CellEnd[hash] = idx + 1;
		}
	}
}

void writeToFile(const std::string& name, uint* cellBegin, uint* cellEnd, uint3 gridSize) {
	std::ofstream outputFile(name, std::ifstream::binary);
	for (int i = 0; i < gridSize.x * gridSize.y * gridSize.z; i++) {
		uint64_t num = cellEnd[i] - cellBegin[i];
//		uint32_t num = i;
		outputFile.write((char*) &num, 4);
//		std::cout << num << ", ";
//		outputFile << num;
	}
	outputFile.close();
}



int main(int argc, char **argv)
{	
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
		
		for (int n = 0; n < pList.size(); n++) {
			auto p = pList[n];
			
			HANDLE_ERROR(cudaMalloc(&d_List[n], p.info.stride * p.info.groupCount));
			HANDLE_ERROR(cudaMemcpy(d_List[n], p.data, p.info.stride * p.info.groupCount, cudaMemcpyHostToDevice));

			long long startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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

			//inc <<<dimGrid, dimBlock >>> (d_List[n], p.info, pSysInfo);
			fillHashGrid << <dimGrid, dimBlock >> > (d_List[n], d_HashList[n], d_IdList[n], p.info, pSysInfo);

//			cudaDeviceSynchronize();

			thrust::sort_by_key(thrust::device_ptr<uint>(d_HashList[n]),
				thrust::device_ptr<uint>(d_HashList[n] + p.info.groupCount * N),
				thrust::device_ptr<uint>(d_IdList[n]));

			setCellPointersOld << <dimGrid, dimBlock >> > (d_HashList[n], d_CellBegin[n], d_CellEnd[n], p.info);
			cudaDeviceSynchronize();

			long long endTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			std::cout << (endTime - startTime) / 1000.0 << std::endl;
			
			if (endTime - startTime < minTime)
				minTime = endTime - startTime;

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
			
			uint* h_CellBegin = new uint[pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z];
			uint* h_CellEnd = new uint[pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z];
			HANDLE_ERROR(cudaMemcpy(h_CellBegin, d_CellBegin[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(h_CellEnd, d_CellEnd[n], sizeof(uint) * pSysInfo.gridSize.x * pSysInfo.gridSize.y * pSysInfo.gridSize.z, cudaMemcpyDeviceToHost));
//			writeToFile("test.raw", h_CellBegin, h_CellEnd, pSysInfo.gridSize);
			
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

	std::cout << minTime << std::endl;
	return 0;
}
