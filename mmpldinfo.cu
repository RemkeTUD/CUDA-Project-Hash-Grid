//
// mmpldinfo.cpp
//
// MMPLDinfo - MegaMol Particle List Data File Information Utility
// Copyright 2014 (C) by S. Grottel, TU Dresden, Germany
// All rights reserved. Alle Rechte vorbehalten.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of S. Grottel, TU Dresden, nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY S. GROTTEL AS IS AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL S. GROTTEL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


//
// compile with:
//   g++ -std=c++0x mmpldinfo.cpp -o mmpldinfo
//

#ifdef _WIN32
#include <tchar.h>
#endif
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>


namespace mmpldinfo {

    /**
     * Prints a simple greeting message
     */
    void print_greeting() {
        std::cout << std::endl
            << "MMPLDinfo - MegaMol Particle List Data File Information Utility" << std::endl
            << "Copyright 2014 (C) by S. Grottel, TU Dresden, Germany" << std::endl
            << "All rights reserved. Alle Rechte vorbehalten." << std::endl
            << std::endl;
    }

    /**
     * Prints information about the specifed mmpld file
     *
     * @param filename The path to the file to print info about
     */
    template<class C>
    void print_fileinfo(const C* filename) {
        const bool list_framedata = true;

        // open file
        std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file");
        }
        // request errors
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        // read and check header
        char magicid[6];
        file.read(magicid, 6);
        if (::memcmp(magicid, "MMPLD", 6) != 0) {
            throw std::runtime_error("File does not seem to be of MMPLD format");
        }
        // read version
        uint16_t version;
        file.read(reinterpret_cast<char*>(&version), 2);
        if (version == 100) {
            std::cout << "    mmpld - version 1.0" << std::endl;
        } else if (version == 101) {
            std::cout << "    mmpld - version 1.1" << std::endl;
        } else {
            std::cerr << "    mmpld - version " << (version / 100) << "." << (version % 100) << std::endl;
            throw std::runtime_error("Unsupported mmpld version encountered");
        }
        // read number of frames
        uint32_t frame_count;
        file.read(reinterpret_cast<char*>(&frame_count), 4);
        std::cout << "Number of frames: " << frame_count << std::endl;
        // read bounding boxes
        float box[6];
        file.read(reinterpret_cast<char*>(box), sizeof(float) * 6);
        std::cout << "Bounding box: (" << box[0] << ", " << box[1] << ", " << box[2] << ", "
             << box[3] << ", " << box[4] << ", " << box[5] << ")" << std::endl;
        file.read(reinterpret_cast<char*>(box), sizeof(float) * 6);
        std::cout << "Clipping box: (" << box[0] << ", " << box[1] << ", " << box[2] << ", "
             << box[3] << ", " << box[4] << ", " << box[5] << ")" << std::endl;
        // abort if no data
        if (frame_count <= 0) {
            throw std::runtime_error("No data");
        }
        // frame index table
        std::vector<uint64_t> frame_table(frame_count + 1);
        file.read(reinterpret_cast<char*>(frame_table.data()), (frame_count + 1) * 8);
        if (static_cast<uint64_t>(file.tellg()) != frame_table[0]) {
            std::cerr << "WRN: dead data trailing head" << std::endl;
        }
        file.seekg(0, std::ios_base::end);
        if (static_cast<uint64_t>(file.tellg()) < frame_table[frame_count]) {
            std::cerr << "WRN: file truncated" << std::endl;
        }
        if (static_cast<uint64_t>(file.tellg()) > frame_table[frame_count]) {
            std::cerr << "WRN: dead data trailing body" << std::endl;
        }
        {
            std::string errmsg;
            for (unsigned int fi = 0; fi < frame_count; fi++) {
                if (frame_table[fi + 1] <= frame_table[fi]) {
                    errmsg += "Frame table corrupted at frame ";
                    errmsg += fi;
                    errmsg += "\n";
                }
            }
            if (!errmsg.empty()) {
                throw std::runtime_error(errmsg);
            }
        }
        // test frames
        uint32_t min_listCnt;
        uint32_t max_listCnt;
        uint64_t min_partCnt;
        uint64_t acc_partCnt(0);
        uint64_t max_partCnt;
        uint64_t frm_partCnt;
        uint64_t lst_partCnt;
        for (uint32_t fi = 0; fi < frame_count; fi++) {
            if (list_framedata) std::cout << "Frame #" << fi;
            file.seekg(frame_table[fi]);
            // number of lists
            uint32_t lists_cnt;
            file.read(reinterpret_cast<char*>(&lists_cnt), 4);
            if (list_framedata) std::cout << " - " << lists_cnt << " list" << ((lists_cnt != 1) ? "s" : "") << std::endl;
            if (fi == 0) {
                min_listCnt = max_listCnt = lists_cnt;
            } else {
                if (min_listCnt > lists_cnt) min_listCnt = lists_cnt;
                if (max_listCnt < lists_cnt) max_listCnt = lists_cnt;
            }
            frm_partCnt = 0;
            for (uint32_t li = 0; li < lists_cnt; li++) {
                // list data format info
                uint8_t vert_type, col_type;
                size_t vrt_size;
                size_t col_size;
                file.read(reinterpret_cast<char*>(&vert_type), 1);
                file.read(reinterpret_cast<char*>(&col_type), 1);
                if (list_framedata) std::cout << "    #" << li << ": ";
                switch (vert_type) {
                    case 1: vrt_size = 12; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZ"; break;
                    case 2: vrt_size = 16; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZR"; break;
                    case 3: vrt_size = 6; if (list_framedata) std::cout << "VERTDATA_SHORT_XYZ"; break;
                    case 0: // falls through
                    default: vrt_size = 0; if (list_framedata) std::cout << "VERTDATA_NONE"; break;
                }
                if (list_framedata) std::cout << ", ";
                if (vert_type == 0) col_type = 0;
                switch (col_type) {
                    case 1: col_size = 3; if (list_framedata) std::cout << "COLDATA_UINT8_RGB"; break;
                    case 2: col_size = 4; if (list_framedata) std::cout << "COLDATA_UINT8_RGBA"; break;
                    case 3: col_size = 4; if (list_framedata) std::cout << "COLDATA_FLOAT_I"; break;
                    case 4: col_size = 12; if (list_framedata) std::cout << "COLDATA_FLOAT_RGB"; break;
                    case 5: col_size = 16; if (list_framedata) std::cout << "COLDATA_FLOAT_RGBA"; break;
                    case 0: // falls through
                    default: col_size = 0; if (list_framedata) std::cout << "COLDATA_NONE"; break;
                }
                if (list_framedata) std::cout << std::endl;
                size_t stride = vrt_size + col_size;
                if (list_framedata) std::cout << "        " << stride << " byte" << ((stride != 1) ? "s" : "") << " per particle" << std::endl;
                float glob_rad(0.05f);
                if ((vert_type == 1) || (vert_type == 3)) {
                    file.read(reinterpret_cast<char*>(&glob_rad), 4);
                    if (list_framedata) std::cout << "        global radius: " << glob_rad << std::endl;
                }
                float col_range[2];
                if (col_type == 0) {
                    uint8_t col[4];
                    file.read(reinterpret_cast<char*>(col), 4);
                    if (list_framedata) std::cout << "        global color: (" << col[0] << ", " << col[1] << ", " << col[2] << ", " << col[3] << ")" << std::endl;
                } else if (col_type == 3) {
                    file.read(reinterpret_cast<char*>(col_range), 8);
                    if (list_framedata) std::cout << "        intensity color range: [" << col_range[0] << ", " << col_range[1] << "]" << std::endl;
                }
                col_range[0] = 0.0f;
                col_range[1] = 1.0f;
                file.read(reinterpret_cast<char*>(&lst_partCnt), 8);
                if (list_framedata) std::cout << "        " << lst_partCnt << " particle" << ((lst_partCnt != 1) ? "s" : "") << std::endl;
                frm_partCnt += lst_partCnt;
                // list data
                // skip (for now; in a later version, we could check for faulty data: particles leaving clipbox or intensity color outside the color range)
                file.seekg(lst_partCnt * stride, std::ios_base::cur);
            }
            if (static_cast<uint64_t>(file.tellg()) != frame_table[fi + 1]) {
                std::cerr << "WRN: trailing data after frame " << fi << std::endl;
            }
            // collect info for particle summary
            if (fi == 0) {
                min_partCnt = max_partCnt = frm_partCnt;
            } else {
                if (min_partCnt > frm_partCnt) min_partCnt = frm_partCnt;
                if (max_partCnt < frm_partCnt) max_partCnt = frm_partCnt;
            }
            acc_partCnt += frm_partCnt;
        }
        acc_partCnt /= frame_count;
        // particle summary
        std::cout << "Data Summary" << std::endl;
        std::cout << "    " << frame_count << " time frame" << ((frame_count != 1) ? "s" : "") << std::endl;
        if (min_listCnt == max_listCnt) {
            std::cout << "    " << min_listCnt << " particle list" << ((min_listCnt != 1) ? "s" : "") << " per frame" << std::endl;
        } else {
            std::cout << "    " << min_listCnt << " .. " << max_listCnt << " particle lists per frame" << std::endl;
        }
        if (min_partCnt == max_partCnt) {
            std::cout << "    " << min_partCnt << " particle" << ((min_partCnt != 1) ? "s" : "") << " per frame" << std::endl;
        } else {
            std::cout << "    " << min_partCnt << " .. " << max_partCnt << " particles per frame" << std::endl;
            std::cout << "    " << acc_partCnt << " on average" << std::endl;
        }
    }

}

/**
 * Application main entry function
 *
 * @param argc number of command line arguments
 * @param argv command line arguments
 * @return 0
 */
#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
    mmpldinfo::print_greeting();
    for (int i = 1; i < argc; i++) {
        std::wcout << "File: " << argv[i] << std::endl;
        std::cout << "========================================" << std::endl;
        try {
            mmpldinfo::print_fileinfo(argv[i]);
        } catch(std::exception ex) {
            std::cerr << "FAILED: " << ex.what() << std::endl;
        } catch(...) {
            std::cerr << "FAILED: unknown exception" << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}
