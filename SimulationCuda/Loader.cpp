#include "Loader.h"

Loader::Loader(const char * filename)
{
	m_file = std::ifstream(filename, std::ifstream::in | std::ifstream::binary);
	if (!m_file.is_open()) {
		throw std::runtime_error("Unable to open m_file");
	}
	// request errors
	m_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	// read and check header
	char magicid[6];
	m_file.read(magicid, 6);
	if (::memcmp(magicid, "MMPLD", 6) != 0) {
		throw std::runtime_error("File does not seem to be of MMPLD format");
	}
	// read version
	uint16_t version;
	m_file.read(reinterpret_cast<char*>(&version), 2);
	if (version == 100) {
		std::cout << "    mmpld - version 1.0" << std::endl;
	}
	else if (version == 101) {
		std::cout << "    mmpld - version 1.1" << std::endl;
	}
	else {
		std::cerr << "    mmpld - version " << (version / 100) << "." << (version % 100) << std::endl;
		throw std::runtime_error("Unsupported mmpld version encountered");
	}
	// read number of frames
	
	m_file.read(reinterpret_cast<char*>(&m_frameCount), 4);
	std::cout << "Number of frames: " << m_frameCount << std::endl;

	// abort if no data
	if (m_frameCount <= 0) {
		throw std::runtime_error("No data");
	}
	/*
	// abort if requested frame invalid
	if (frameId < 0 || frameId >= m_frameCount) {
		throw std::runtime_error("Invalid Frame requested");
	}
	*/

	// read bounding boxes
	float box[6];
	m_file.read(reinterpret_cast<char*>(box), sizeof(float) * 6);
	m_lowBBox.x = box[0];
	m_lowBBox.y = box[1];
	m_lowBBox.z = box[2];
	m_highBBox.x = box[3];
	m_highBBox.y = box[4];
	m_highBBox.z = box[5];

	std::cout << "Bounding box: (" << box[0] << ", " << box[1] << ", " << box[2] << ", "
		<< box[3] << ", " << box[4] << ", " << box[5] << ")" << std::endl;

	m_file.read(reinterpret_cast<char*>(box), sizeof(float) * 6);
	m_lowCBox.x = box[0];
	m_lowCBox.y = box[1];
	m_lowCBox.z = box[2];
	m_highCBox.x = box[3];
	m_highCBox.y = box[4];
	m_highCBox.z = box[5];

	std::cout << "Clipping box: (" << box[0] << ", " << box[1] << ", " << box[2] << ", "
		<< box[3] << ", " << box[4] << ", " << box[5] << ")" << std::endl;

	// frame index table
	m_frameTable = std::vector<uint64_t>(m_frameCount + 1);
	m_file.read(reinterpret_cast<char*>(m_frameTable.data()), (m_frameCount + 1) * 8);
	if (static_cast<uint64_t>(m_file.tellg()) != m_frameTable[0]) {
		std::cerr << "WRN: dead data trailing head" << std::endl;
	}
	m_file.seekg(0, std::ios_base::end);
	if (static_cast<uint64_t>(m_file.tellg()) < m_frameTable[m_frameCount]) {
		std::cerr << "WRN: m_file truncated" << std::endl;
	}
	if (static_cast<uint64_t>(m_file.tellg()) > m_frameTable[m_frameCount]) {
		std::cerr << "WRN: dead data trailing body" << std::endl;
	}
	{
		std::string errmsg;
		for (unsigned int fi = 0; fi < m_frameCount; fi++) {
			if (m_frameTable[fi + 1] <= m_frameTable[fi]) {
				errmsg += "Frame table corrupted at frame ";
				errmsg += fi;
				errmsg += "\n";
			}
		}
		if (!errmsg.empty()) {
			throw std::runtime_error(errmsg);
		}
	}
}

Loader::~Loader()
{
	m_file.close();
}

std::vector<ParticleList> Loader::getFrame(uint32_t frameId)
{
	// read frame
	m_file.seekg(m_frameTable[frameId]);
	// number of lists
	uint32_t listsCnt;
	m_file.read(reinterpret_cast<char*>(&listsCnt), 4);

	std::vector<ParticleList> pFrame;
	for (uint32_t li = 0; li < listsCnt; li++) {
		ParticleInfo pInfo;
		uint8_t vertType, colType;

		m_file.read(reinterpret_cast<char*>(&vertType), 1);
		m_file.read(reinterpret_cast<char*>(&colType), 1);
		std::cout << "    #" << frameId << ": ";
		switch (vertType) {
		case 1: pInfo.stride = 12; std::cout << "VERTDATA_FLOAT_XYZ"; break;
		case 2: pInfo.stride = 16; std::cout << "VERTDATA_FLOAT_XYZR"; break;
		case 3: pInfo.stride = 6; std::cout << "VERTDATA_SHORT_XYZ"; break;
		case 0: // falls through
		default: pInfo.stride = 0; std::cout << "VERTDATA_NONE"; break;
		}
		std::cout << ", ";
		switch (colType) {
		case 1: pInfo.stride += 3; std::cout << "COLDATA_UINT8_RGB"; break;
		case 2: pInfo.stride += 4; std::cout << "COLDATA_UINT8_RGBA"; break;
		case 3: pInfo.stride += 4; std::cout << "COLDATA_FLOAT_I"; break;
		case 4: pInfo.stride += 12; std::cout << "COLDATA_FLOAT_RGB"; break;
		case 5: pInfo.stride += 16; std::cout << "COLDATA_FLOAT_RGBA"; break;
		case 0: // falls through
		default: std::cout << "COLDATA_NONE"; break;
		}
		std::cout << std::endl;

		if ((vertType == 1) || (vertType == 3)) {
			m_file.read(reinterpret_cast<char*>(&pInfo.groupRadius), 4);
			std::cout << "        global radius: " << pInfo.groupRadius << std::endl;
		}
		if (colType == 0) {
			uint8_t col[4];
			m_file.read(reinterpret_cast<char*>(col), 4);
			std::cout << "        global color: (" << col[0] << ", " << col[1] << ", " << col[2] << ", " << col[3] << ")" << std::endl;
		}
		else if (colType == 3) {
			float col_range[2];
			m_file.read(reinterpret_cast<char*>(col_range), 8);
			std::cout << "        intensity color range: [" << col_range[0] << ", " << col_range[1] << "]" << std::endl;
		}
		m_file.read(reinterpret_cast<char*>(&pInfo.groupCount), 8);
		std::cout << "        " << pInfo.groupCount << " particle" << ((pInfo.groupCount != 1) ? "s" : "") << std::endl;
		
		char* data = new char[pInfo.groupCount * pInfo.stride];

		m_file.read(data, pInfo.groupCount * pInfo.stride);
		ParticleList pList;
		pList.info = pInfo;
		pList.data = data;
		pFrame.push_back(pList);
	}
	return pFrame;
}

PSystemInfo Loader::calcBSystemInfo(uint3 gridSize)
{
	PSystemInfo pSysInfo;
	pSysInfo.gridSize = gridSize;
	pSysInfo.worldOrigin = m_lowBBox;
	pSysInfo.cellSize.x = (m_highBBox.x - m_lowBBox.x) / gridSize.x;
	pSysInfo.cellSize.y = (m_highBBox.y - m_lowBBox.y) / gridSize.y;
	pSysInfo.cellSize.z = (m_highBBox.z - m_lowBBox.z) / gridSize.z;
	std::cout << "CellSize: " << pSysInfo.cellSize.x << ", " << pSysInfo.cellSize.y << ", " << pSysInfo.cellSize.z << std::endl;
	return pSysInfo;
}

PSystemInfo Loader::calcBSystemInfo(float3 cellSize)
{
	PSystemInfo pSysInfo;
	pSysInfo.cellSize = cellSize;
	pSysInfo.worldOrigin = m_lowBBox;
	pSysInfo.gridSize.x = ceil((m_highBBox.x - m_lowBBox.x) / cellSize.x);
	pSysInfo.gridSize.y = ceil((m_highBBox.y - m_lowBBox.y) / cellSize.y);
	pSysInfo.gridSize.z = ceil((m_highBBox.z - m_lowBBox.z) / cellSize.z);
	return pSysInfo;
}


PSystemInfo Loader::calcBSystemInfo(std::vector<ParticleList> pList)
{
	float maxRadius = 0;
	for (int a = 0; a < pList.size(); a++){
		if (pList[a].info.groupRadius != -1) {
			float cs = pList[a].info.groupRadius * 2;
			if (cs > maxRadius)
				maxRadius = cs;
		}

		else {
			for (int i = 0; i < pList[a].info.groupCount; i++) {
				float4* pos = (float4*) (pList[a].data + i * pList[a].info.stride);
				if (pos->w > maxRadius)
					maxRadius = pos->w;
			}
		}
	}

	float3 cellSize;
	cellSize.x = maxRadius * 2.0f;
	cellSize.y = maxRadius * 2.0f;
	cellSize.z = maxRadius * 2.0f;

	return calcBSystemInfo(cellSize);
}
