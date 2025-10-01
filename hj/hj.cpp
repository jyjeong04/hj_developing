#include "hj.hpp"
#include "param.hpp"
#include "datagen.cpp"
#include "util.hpp"

#include "cl.hpp"
#include "device_picker.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
 
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cctype>
#include <ostream>
#include <vector>
#include <unordered_map>

 

static std::vector<JoinedTuple> run_standard_hash_join(const std::vector<Tuple>& R,
        const std::vector<Tuple>& S) {
    // Build hash table from R: key -> list of R rids
    std::unordered_map<uint32_t, std::vector<uint32_t>> rIndex;
    rIndex.reserve(static_cast<size_t>(R_LENGTH) * 2);
    for(const auto &t : R) {
        rIndex[t.key].push_back(t.rid);
    }

    // Probe with S and emit joins
    std::vector<JoinedTuple> out;
    // Rough reservation heuristic to reduce reallocations
    out.reserve(static_cast<size_t>(S_LENGTH) / 4);
    for(const auto &s : S) {
        auto it = rIndex.find(s.key);
        if(it == rIndex.end()) continue;
        const std::vector<uint32_t>& rids = it->second;
        for(size_t i = 0; i < rids.size(); i++) {
            JoinedTuple jt;
            jt.key = s.key;
            jt.ridR = rids[i];
            jt.ridS = s.rid;
            out.push_back(jt);
        }
    }
    return out;
}

uint32_t hash(uint32_t key) {
    return (key * 2654435769U) % (R_LENGTH * 2);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    std::vector<BucketHeader> bucketList(R_LENGTH * 2);
    // std::vector<Tuple> R = RGenerator();
    // std::vector<Tuple> S = SGenerator(R);
    std::vector<Tuple> R(R_LENGTH);
    std::vector<Tuple> S(S_LENGTH);
    int i;
    for(i = 0; i < R_LENGTH; i++) {
        R[i].key = rand() % R_LENGTH;
        R[i].rid = rand() % 1024;
    }
    for(i = 0; i < S_LENGTH; i++) {
        S[i].key = rand() % R_LENGTH;
        S[i].rid = rand() % 1024;
    }
    std::vector<JoinedTuple> res;
    res.reserve(S_LENGTH / 4);

    util::Timer timer;

    timer.reset();
    for(i = 0; i < R_LENGTH; i++) {
        // b1: compute hash bucket number
        Tuple &tmpTuple = R[i];
        uint32_t id = hash(tmpTuple.key);
        // b2: visit the hash bucket header
        BucketHeader &tmpHeader = bucketList[id];
        // b3: visit the hash key lists and create a key header if necessary
        int j = 0;
        bool found = false;
        for(j = 0; j < tmpHeader.totalNum; j++) {
            if(tmpTuple.key == tmpHeader.keyList[j].key) {
                found = true;
                break;
            }
        }
        tmpHeader.totalNum++;
        
        if(!found) {
            KeyHeader newKey;
            newKey.key = tmpTuple.key;
            tmpHeader.keyList.push_back(newKey);
        }
        
        // b4: insert the rid into the rid list
        tmpHeader.keyList[j].ridList.push_back(tmpTuple.rid);
    }
    for(i = 0; i < S_LENGTH; i++) {
        // p1: compute hash bucket number
        Tuple &tmpTuple = S[i];
        uint32_t id = hash(tmpTuple.key);
        // p2: visit the hash bucket header
        BucketHeader &tmpHeader = bucketList[id];
        // p3: visit the hash key lists
        int j = 0;
        bool found = false;
        for(j = 0; j < tmpHeader.totalNum; j++) {
            if(tmpTuple.key == tmpHeader.keyList[j].key) {
                found = true;
                break;
            }
        }

        // p4: visit the matching build tuple to compare keys and produce output tuple
        if(found) {
            for(int h = 0; h < tmpHeader.keyList[j].ridList.size(); h++) {
                JoinedTuple t;
                t.key = tmpTuple.key;
                t.ridR = tmpHeader.keyList[j].ridList[h];
                t.ridS = tmpTuple.rid;
                res.push_back(t);
            }
        }
    }

    std::cout << timer.getTimeMilliseconds() << "ms" << std::endl;

    // Run standard hash join and verify against hash-join result
    util::Timer timer2;
    timer2.reset();
    std::vector<JoinedTuple> stdRes = run_standard_hash_join(R, S);
    double stdMs = timer2.getTimeMilliseconds();
    std::cout << "std_join: " << stdRes.size() << " tuples, " << stdMs << "ms\n";

//===================== OpenCL Join ==========================

    try
    {
        cl_uint deviceIndex = 0;
        
        // Simple argument parsing: ./hj 1 or --device 1
        if (argc >= 2) {
            // Check if it's just a number (simple format)
            if (argc == 2 && isdigit(argv[1][0])) {
                deviceIndex = atoi(argv[1]);
            } else {
                // Use the original parseArguments for --device format
                parseArguments(argc, argv, &deviceIndex);
            }
        }

        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        if(deviceIndex >= numDevices) {
            std::cout << "Invalid device index";
            return EXIT_FAILURE;
        }
        
        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL Device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // Load kernel source files using util::loadProgram
        std::cout << "Loading kernel source files..." << std::endl;
        std::string b1_source = util::loadProgram("b1.cl");
        std::string b2_source = util::loadProgram("b2.cl");
        std::string b3_source = util::loadProgram("b3.cl");
        std::string b4_source = util::loadProgram("b4.cl");
        std::string p1_source = util::loadProgram("p1.cl");
        std::string p2_source = util::loadProgram("p2.cl");
        std::string p3_source = util::loadProgram("p3.cl");
        std::string p4_source = util::loadProgram("p4.cl");

        // Create programs and kernels
        cl::Program b1_program(context, b1_source);
        cl::Program b2_program(context, b2_source);
        cl::Program b3_program(context, b3_source);
        cl::Program b4_program(context, b4_source);
        cl::Program p1_program(context, p1_source);
        cl::Program p2_program(context, p2_source);
        cl::Program p3_program(context, p3_source);
        cl::Program p4_program(context, p4_source);

    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
































//===================== OpenCL Join ==========================

    // Compare by total count
    bool pass = (res.size() == stdRes.size());

    // Compare by per-key counts
    if(pass) {
        std::unordered_map<uint32_t, uint64_t> resKeyCount;
        std::unordered_map<uint32_t, uint64_t> stdKeyCount;
        resKeyCount.reserve(R_LENGTH);
        stdKeyCount.reserve(R_LENGTH);
        for(const auto &jt : res) resKeyCount[jt.key]++;
        for(const auto &jt : stdRes) stdKeyCount[jt.key]++;
        if(resKeyCount.size() != stdKeyCount.size()) {
            pass = false;
        } else {
            for(const auto &kv : resKeyCount) {
                auto it = stdKeyCount.find(kv.first);
                if(it == stdKeyCount.end() || it->second != kv.second) { pass = false; break; }
            }
        }
    }

    std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
    return 0;

//     // Print a few sample results (up to 20)
//     std::cout << "RESULT (first up to 20 tuples)\n";
//     int printN = static_cast<int>(std::min<size_t>(20, res.size()));
//     for(int k = 0; k < printN; k++) {
//         std::cout << "(key=" << res[k].key
//                   << ", ridR=" << res[k].ridR
//                   << ", ridS=" << res[k].ridS << ")\n";
//     }
}

