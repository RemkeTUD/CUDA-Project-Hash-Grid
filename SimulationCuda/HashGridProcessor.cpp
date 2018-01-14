// HashGridProcessor.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "Loader.h"

int3 calcGridPos(float3 p, PSystemInfo pSysInfo)
{
	int3 gridPos;
	gridPos.x = floor((p.x - pSysInfo.worldOrigin.x) / pSysInfo.cellSize.x);
	gridPos.y = floor((p.y - pSysInfo.worldOrigin.y) / pSysInfo.cellSize.y);
	gridPos.z = floor((p.z - pSysInfo.worldOrigin.z) / pSysInfo.cellSize.z);
	return gridPos;
}

uint calcGridHash(int3 gridPos, PSystemInfo pSysInfo)
{
	return gridPos.z * pSysInfo.gridSize.y * pSysInfo.gridSize.x + gridPos.y * pSysInfo.gridSize.x + gridPos.x;
}

void writeToFile(const std::string& name, std::vector<std::vector<uint>> grid, uint3 gridSize) {
	std::ofstream outputFile(name, std::ifstream::binary);
	for (int i = 0; i < gridSize.x * gridSize.y * gridSize.z; i++) {
		uint64_t num = grid[i].size();
		//		uint32_t num = i;
		outputFile.write((char*)&num, 4);
		//		std::cout << num << ", ";
		//		outputFile << num;
	}
	outputFile.close();
}

int main()
{
	Loader loader("exp2mill.mmpld");

	auto pList = loader.getFrame(20);

	uint3 gridSize;
	gridSize.x = 32;
	gridSize.y = 320;
	gridSize.z = 32;
	PSystemInfo pSysInfo = loader.calcBSystemInfo(gridSize);
	
	std::cout << "Grid Size: (" << pSysInfo.gridSize.x << ", " << pSysInfo.gridSize.y << ", " << pSysInfo.gridSize.z << ") Origin: (" << pSysInfo.worldOrigin.x << ", " << pSysInfo.worldOrigin.y << ", " << pSysInfo.worldOrigin.z << ") Cell Size: (" << pSysInfo.cellSize.x << ", " << pSysInfo.cellSize.y << ", " << pSysInfo.cellSize.z << ")" << std::endl;

	std::vector<std::vector<uint>> grid(gridSize.x * gridSize.y * gridSize.z);

	for (int n = 0; n < pList.size(); n++) {
		auto p = pList[n];
		for(int i = 0; i < p.info.groupCount; i++)
			grid[calcGridHash(calcGridPos(*(float3*)(p.data + i * p.info.stride), pSysInfo),pSysInfo)].push_back(n);

	}

	writeToFile("testFile.raw", grid, pSysInfo.gridSize);

    return 0;
}

