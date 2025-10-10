#include "hj.hpp"
#include "param.hpp"
#include "datagen.cpp"
#include "util.hpp"

#include "cl.hpp"
#include "device_picker.hpp"
#include <CL/cl.h>
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
    return (key * 2654435769U) % (BUCKET_HEADER_NUMBER);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    std::vector<BucketHeader> bucketList(BUCKET_HEADER_NUMBER);
    // std::vector<Tuple> R = RGenerator();
    // std::vector<Tuple> S = SGenerator(R);
    std::vector<Tuple> R(R_LENGTH);
    std::vector<Tuple> S(S_LENGTH);
    int i;
    for(i = 0; i < R_LENGTH; i++) {
        R[i].key = rand() % R_LENGTH + 1;
        R[i].rid = rand() % 1024;
    }
    for(i = 0; i < S_LENGTH; i++) {
        S[i].key = rand() % R_LENGTH + 1;
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
        if(!tmpHeader.totalNum) continue;
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

// ===================== OpenCL Join ==========================

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

        // Create programs and kernels
        cl::Program b1_program(context, util::loadProgram("b1.cl"), true);
        cl::Program b2_program(context, util::loadProgram("b2.cl"), true);
        cl::Program b3_program(context, util::loadProgram("b3.cl"), true);
        cl::Program b4_program(context, util::loadProgram("b4.cl"), true);
        cl::Program p1_program(context, util::loadProgram("p1.cl"), true);
        cl::Program p2_program(context, util::loadProgram("p2.cl"), true);
        cl::Program p3_program(context, util::loadProgram("p3.cl"), true);
        cl::Program p4_program(context, util::loadProgram("p4.cl"), true);

        cl::make_kernel<cl::Buffer, cl::Buffer> b1(b1_program, "b1");
        cl::make_kernel<cl::Buffer, cl::Buffer> b2(b2_program, "b2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(b3_program, "b3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(b4_program, "b4");
        cl::make_kernel<cl::Buffer, cl::Buffer> p1(p1_program, "p1");
        cl::make_kernel<cl::Buffer, cl::Buffer> p2(p2_program, "p2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> p3(p3_program, "p3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> p4(p4_program, "p4");

        std::vector<uint32_t> R_keys(R_LENGTH), R_rids(R_LENGTH), S_keys(S_LENGTH), S_rids(S_LENGTH);
        for(int i = 0; i < R_LENGTH; i++) {
            R_keys[i] = R[i].key;
            R_rids[i] = R[i].rid;
        }
        for(int i = 0; i < S_LENGTH; i++) {
            S_keys[i] = S[i].key;
            S_rids[i] = S[i].rid;
        }

        // buffer init
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        cl::Buffer R_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * R_LENGTH, &R_rids[0]);
        cl::Buffer S_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_rids[0]);
        
        // b1
        cl::Buffer R_bucket_ids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * R_LENGTH);

        // b2
        cl::Buffer R_bucket_totalNum_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER);

        // b3
        cl::Buffer bucket_keys_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer bucket_key_counts_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH);

        // b4
        cl::Buffer bucket_key_rids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY);
        cl::Buffer bucket_key_rid_counts_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);

        // p1
        cl::Buffer S_bucket_ids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH);
        
        // p2

        // p3
        cl::Buffer S_key_indices_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * S_LENGTH);
        cl::Buffer S_match_found_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH);

        // p4
        cl::Buffer result_key_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH / 4);
        cl::Buffer result_rid_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH / 4);
        cl::Buffer result_sid_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH / 4);
        cl::Buffer result_idx_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t));

        std::vector<uint32_t> bucket_totalNumcounts(BUCKET_HEADER_NUMBER, 0);
        std::vector<uint32_t> bucket_key_count

        



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
































//===================== OpenCL Join End ==========================

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

