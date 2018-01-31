#include "HashGrid.cuh"


void writeToFileCPU(const std::string& name, std::vector<std::vector<uint>> grid, uint3 gridSize) {
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



float benchmarkPListGPU(ParticleList p, PSystemInfo pSysInfo, int iterations = 1) {
	float minTime = 100000000.0f;
	for (int i = 0; i < iterations; i++) {
		HashGrid hGrid = HashGrid(p, pSysInfo);

//		Particle particle;
//		particle.pos = make_float3(60, 600, 60);
//		particle.radius = 60;

//		std::cout << getNumberOfNeighboursGPUOld(d_List, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle, isAligned) << std::endl;
//		std::cout << getNumberOfNeighboursGPU(d_List, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle, isAligned) << std::endl;
//		std::cout << getNumberOfNeighboursCPU(p.data, d_CellBegin, d_CellEnd, d_IdList, p.info, pSysInfo, particle) << std::endl;

		if (hGrid.initHashGridTime < minTime)
			minTime = hGrid.initHashGridTime;


	}
	std::cout << "GPU: Particle count: " << p.info.groupCount << " Time: " << minTime << std::endl;
	return minTime;
}

void benchmarkPListGPU(ParticleList p, PSystemInfo pSysInfo, float& minTime, float& minCopyTime, float& minAllocTime, float& minKernelTime, int iterations = 1) {
	minTime = std::numeric_limits<float>::max();
	minCopyTime = std::numeric_limits<float>::max();
	minAllocTime = std::numeric_limits<float>::max();
	minKernelTime = std::numeric_limits<float>::max();
	for (int i = 0; i < iterations; i++) {
		HashGrid hGrid = HashGrid(p, pSysInfo);

		if (hGrid.initHashGridTime < minTime)
			minTime = hGrid.initHashGridTime;

		if (hGrid.copyDataTime < minCopyTime)
			minCopyTime = hGrid.copyDataTime;

		if (hGrid.allocDataTime < minAllocTime)
			minAllocTime = hGrid.allocDataTime;

		if (hGrid.kernelTime < minKernelTime)
			minKernelTime = hGrid.kernelTime;


	}
	std::cout << "GPU: Particle count: " << p.info.groupCount << " Time: " << minTime << std::endl;
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
	

	Loader loader("laser.00080.chkpt.density.mmpld");

	auto pLists = loader.getFrame(0);


	
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
//	loader.isInBBox(pLists, pSysInfo);
//	ParticleList pList = reduceParticles(pLists[0], 20000000);
//	benchmarkPListGPU(pLists[0], pSysInfo, 10);
//	benchmarkPListCPU(pLists[0], pSysInfo, 10);
	
//	HashGrid hGrid = HashGrid(pLists[0], pSysInfo);
//	hGrid.writeToRawFile("exp2mill.raw");


	std::ofstream outputFile("benchmark.csv");
	outputFile << "Dimension; GPU; CPU\n";
	
	for (uint i = 0; i <= 9; i++) {
		uint3 gridSize;
		gridSize.x = std::pow(2, i);
		gridSize.y = std::pow(2, i);
		gridSize.z = std::pow(2, i);
		PSystemInfo pSysInfo = loader.calcBSystemInfo(gridSize);
		outputFile << gridSize.x << "x" << gridSize.x << "x" << gridSize.x << ";";
		outputFile << std::round(benchmarkPListGPU(pLists[0], pSysInfo, 10)) << ";";
		outputFile << std::round(benchmarkPListCPU(pLists[0], pSysInfo, 10) / 1000.0f) << "\n";
		
		/*
		float time;
		Particle p;
//		p.pos = make_float3(pSysInfo.gridSize.x * 0.5f * pSysInfo.cellSize.x + pSysInfo.worldOrigin.x, pSysInfo.gridSize.y * 0.5f * pSysInfo.cellSize.y + pSysInfo.worldOrigin.y, pSysInfo.gridSize.z * 0.5f * pSysInfo.cellSize.z + pSysInfo.worldOrigin.z);
//		p.radius = i * pSysInfo.cellSize.x;
		p.pos = make_float3(pSysInfo.worldOrigin.x + pSysInfo.gridSize.x * pSysInfo.cellSize.x * i / 30.0f, pSysInfo.gridSize.y * 0.5f * pSysInfo.cellSize.y + pSysInfo.worldOrigin.y, pSysInfo.gridSize.z * 0.5f * pSysInfo.cellSize.z + pSysInfo.worldOrigin.z);
		p.radius = 60;
		std::cout << "Radius: " << p.radius << std::endl;
		std::cout << p.pos.x << ", " << p.pos.y << ", " << p.pos.z << std::endl;
		outputFile << std::round(p.pos.x) << ";";
		
		uint nGPU = hGrid.getNumberOfNeighboursGPU(p, time, 10);
		outputFile << std::round(time * 1000) << ";";
		uint nCPU = hGrid.getNumberOfNeighboursCPU(p, time, 10);
		outputFile << std::round(time * 1000) << "\n";
		std::cout << "GPU: " << nGPU << "; CPU: " << nCPU << std::endl;
		*/
	}
	outputFile.close();
	

//	HashGrid hGrid = HashGrid(pLists[0], pSysInfo);
//	hGrid.writeToRawFile("60mill.raw");
	return 0;
}
