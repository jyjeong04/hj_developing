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
    
    // Parse arguments: ./hj [device_index] [--skip-cpu] [--skip-std] [--help]
    bool run_cpu_join = false;
    bool run_std_join = false;
    
    for(int arg_i = 1; arg_i < argc; arg_i++) {
        if(strcmp(argv[arg_i], "--cpu") == 0) {
            run_cpu_join = true;
        } else if(strcmp(argv[arg_i], "--std") == 0) {
            run_std_join = true;
        } else if(strcmp(argv[arg_i], "--help") == 0 || strcmp(argv[arg_i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [device_index] [options]\n"
                      << "Options:\n"
                      << "  --cpu     Run CPU hash join\n"
                      << "  --std     Run standard hash join\n"
                      << "  --help, -h     Show this help message\n"
                      << "\nExample:\n"
                      << "  " << argv[0] << " 0              # Run all joins on GPU device 0\n"
                      << "  " << argv[0] << " 0 --skip-cpu   # Run only standard and OpenCL joins\n"
                      << "  " << argv[0] << " 1 --skip-std   # Run only CPU and OpenCL joins on CPU device\n";
            return 0;
        }
    }
    
    std::vector<BucketHeader> bucketList(BUCKET_HEADER_NUMBER);
    
    // Generate datasets using datagen.cpp functions
    std::vector<Tuple> R = RGenerator();
    std::vector<Tuple> S = SGenerator(R);
    
    std::vector<JoinedTuple> res;

    util::Timer timer;

    // CPU-based Hash Join
    if(run_cpu_join) {
        std::cout << "=== CPU Hash Join ===\n";
        timer.reset();
        for(int i = 0; i < R_LENGTH; i++) {
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
            if(!found) {
                KeyHeader newKey;
                tmpHeader.totalNum++;
                newKey.key = tmpTuple.key;
                tmpHeader.keyList.push_back(newKey);
            }
            
            // b4: insert the rid into the rid list
            tmpHeader.keyList[j].ridList.push_back(tmpTuple.rid);
        }
        for(int i = 0; i < S_LENGTH; i++) {
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

        std::cout << "CPU Join: " << res.size() << " tuples, " << timer.getTimeMilliseconds() << "ms" << std::endl;
    }

    // Run standard hash join and verify against hash-join result
    std::vector<JoinedTuple> stdRes;
    if(run_std_join) {
        std::cout << "\n=== Standard Hash Join ===" << std::endl;
        util::Timer timer2;
        timer2.reset();
        stdRes = run_standard_hash_join(R, S);
        double stdMs = timer2.getTimeMilliseconds();
        std::cout << "Standard Join: " << stdRes.size() << " tuples, " << stdMs << "ms\n";
    }

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
        cl::Program program(context, util::loadProgram("hj.cl"), true);

        cl::make_kernel<cl::Buffer, cl::Buffer> b1(program, "b1");
        cl::make_kernel<cl::Buffer, cl::Buffer> b2(program, "b2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(program, "b3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(program, "b4");
        cl::make_kernel<cl::Buffer, cl::Buffer> p1(program, "p1");
        cl::make_kernel<cl::Buffer, cl::Buffer> p2(program, "p2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> p3(program, "p3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> p4(program, "p4");

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
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        cl::Buffer R_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * R_LENGTH, &R_rids[0]);
        cl::Buffer S_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_rids[0]);
        
        // b1
        cl::Buffer R_bucket_ids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * R_LENGTH);

        // b2
        cl::Buffer bucket_total_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER);

        // b3
        cl::Buffer bucket_keys_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH);

        // b4
        cl::Buffer bucket_key_rids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY);

        // p1
        cl::Buffer S_bucket_ids_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH);
        
        // p2

        // p3
        cl::Buffer S_key_indices_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * S_LENGTH);
        cl::Buffer S_match_found_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH);

        // p4 - Pre-allocate large buffer: S_LENGTH * MAX_RIDS_PER_KEY
        // Each S tuple gets MAX_RIDS_PER_KEY slots - NO ATOMIC OPERATIONS NEEDED
        size_t max_result_size = (size_t)S_LENGTH * MAX_RIDS_PER_KEY;
        cl::Buffer result_key_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * max_result_size);
        cl::Buffer result_rid_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * max_result_size);
        cl::Buffer result_sid_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * max_result_size);
        cl::Buffer result_count_buf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t) * S_LENGTH);

        std::vector<uint32_t> bucket_totalNumcounts(BUCKET_HEADER_NUMBER, 0);

        // Initialize buffers to zero
        queue.enqueueWriteBuffer(bucket_total_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER, &bucket_totalNumcounts[0]);
        
        // Zero-initialize large buffers
        std::vector<uint32_t> bucket_key_rids_init(BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY, 0xffffffffu);
        std::vector<uint32_t> bucket_keys_init(BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0xffffffffu);
        std::vector<uint32_t> result_count_init(S_LENGTH, 0);
        queue.enqueueWriteBuffer(bucket_key_rids_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY, &bucket_key_rids_init[0]);
        queue.enqueueWriteBuffer(bucket_keys_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, &bucket_keys_init[0]);
        queue.enqueueWriteBuffer(result_count_buf, CL_TRUE, 0, sizeof(uint32_t) * S_LENGTH, &result_count_init[0]);

        util::Timer opencl_timer;
        opencl_timer.reset();

        // Build Phase
        std::cout << "\n=== OpenCL Build Phase ===" << std::endl;
        
        util::Timer step_timer;
        
        // b1: compute hash bucket number
        step_timer.reset();
        b1(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_keys_buf, R_bucket_ids_buf);
        queue.finish();
        double b1_time = step_timer.getTimeMilliseconds();
        std::cout << "  b1 (compute hash): " << b1_time << " ms" << std::endl;
        
        // b2: update bucket header
        step_timer.reset();
        b2(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_bucket_ids_buf, bucket_total_buf);
        queue.finish();
        double b2_time = step_timer.getTimeMilliseconds();
        std::cout << "  b2 (bucket count): " << b2_time << " ms" << std::endl;
        
        // b3: manage key lists
        step_timer.reset();
        b3(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), 
            R_keys_buf, R_bucket_ids_buf, bucket_keys_buf, key_indices_buf);
        queue.finish();
        double b3_time = step_timer.getTimeMilliseconds();
        std::cout << "  b3 (key management): " << b3_time << " ms" << std::endl;
        
        // b4: insert record ids
        step_timer.reset();
        b4(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), 
            R_rids_buf, R_bucket_ids_buf, key_indices_buf, 
            bucket_key_rids_buf);
        queue.finish();
        double b4_time = step_timer.getTimeMilliseconds();
        std::cout << "  b4 (insert rids): " << b4_time << " ms" << std::endl;

        double build_total = b1_time + b2_time + b3_time + b4_time;
        std::cout << "Build Phase Total: " << build_total << " ms" << std::endl;

        // Probe Phase
        std::cout << "\n=== OpenCL Probe Phase ===" << std::endl;
        
        // p1: compute hash bucket number
        step_timer.reset();
        p1(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_keys_buf, S_bucket_ids_buf);
        queue.finish();
        double p1_time = step_timer.getTimeMilliseconds();
        std::cout << "  p1 (compute hash): " << p1_time << " ms" << std::endl;
        
        // p2: check bucket validity
        step_timer.reset();
        p2(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_bucket_ids_buf, bucket_total_buf);
        queue.finish();
        double p2_time = step_timer.getTimeMilliseconds();
        std::cout << "  p2 (bucket check): " << p2_time << " ms" << std::endl;
        
        // p3: search key lists
        step_timer.reset();
        p3(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), 
            S_keys_buf, S_bucket_ids_buf, bucket_keys_buf, 
            S_key_indices_buf, S_match_found_buf);
        queue.finish();
        double p3_time = step_timer.getTimeMilliseconds();
        std::cout << "  p3 (key search): " << p3_time << " ms" << std::endl;

        // p4: join matching records (NO ATOMIC OPERATIONS!)
        step_timer.reset();
        p4(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), 
            S_keys_buf, S_rids_buf, S_key_indices_buf, S_match_found_buf,
            bucket_key_rids_buf, S_bucket_ids_buf,
            result_key_buf, result_rid_buf, result_sid_buf, result_count_buf);
        queue.finish();
        double p4_time = step_timer.getTimeMilliseconds();
        std::cout << "  p4 (join output): " << p4_time << " ms" << std::endl;

        double probe_total = p1_time + p2_time + p3_time + p4_time;
        std::cout << "Probe Phase Total: " << probe_total << " ms" << std::endl;

        double opencl_time = opencl_timer.getTimeMilliseconds();
        std::cout << "\nOpenCL Join Total: " << opencl_time << " ms" << std::endl;

        // Read back result counts directly from GPU
        std::vector<uint32_t> result_counts(S_LENGTH);
        queue.enqueueReadBuffer(result_count_buf, CL_TRUE, 0, sizeof(uint32_t) * S_LENGTH, &result_counts[0]);
        
        // Calculate total number of results
        uint32_t num_results = 0;
        for(uint32_t i = 0; i < S_LENGTH; i++) {
            num_results += result_counts[i];
        }
        
        std::cout << "OpenCL produced " << num_results << " joined tuples" << std::endl;

        if(num_results > 0) {
            // Read the sparse result buffers
            std::vector<uint32_t> sparse_keys(max_result_size);
            std::vector<uint32_t> sparse_rids(max_result_size);
            std::vector<uint32_t> sparse_sids(max_result_size);
            
            queue.enqueueReadBuffer(result_key_buf, CL_TRUE, 0, sizeof(uint32_t) * max_result_size, &sparse_keys[0]);
            queue.enqueueReadBuffer(result_rid_buf, CL_TRUE, 0, sizeof(uint32_t) * max_result_size, &sparse_rids[0]);
            queue.enqueueReadBuffer(result_sid_buf, CL_TRUE, 0, sizeof(uint32_t) * max_result_size, &sparse_sids[0]);
            
            // Compact results: remove empty slots
            std::vector<uint32_t> result_keys;
            std::vector<uint32_t> result_rids;
            std::vector<uint32_t> result_sids;
            result_keys.reserve(num_results);
            result_rids.reserve(num_results);
            result_sids.reserve(num_results);
            
            for(uint32_t i = 0; i < S_LENGTH; i++) {
                uint32_t count = result_counts[i];
                if(count > 0) {
                    uint32_t base_offset = i * MAX_RIDS_PER_KEY;
                    for(uint32_t j = 0; j < count; j++) {
                        result_keys.push_back(sparse_keys[base_offset + j]);
                        result_rids.push_back(sparse_rids[base_offset + j]);
                        result_sids.push_back(sparse_sids[base_offset + j]);
                    }
                }
            }
            
            // Convert to JoinedTuple format
            std::vector<JoinedTuple> opencl_res;
            opencl_res.reserve(num_results);
            for(uint32_t i = 0; i < num_results; i++) {
                JoinedTuple jt;
                jt.key = result_keys[i];
                jt.ridR = result_rids[i];
                jt.ridS = result_sids[i];
                opencl_res.push_back(jt);
            }

            // Compare OpenCL result with Standard join result
            if(run_std_join) {
                bool opencl_pass = (opencl_res.size() == stdRes.size());
                if(opencl_pass) {
                    std::unordered_map<uint32_t, uint64_t> openclKeyCount;
                    std::unordered_map<uint32_t, uint64_t> stdKeyCount;
                    openclKeyCount.reserve(R_LENGTH);
                    stdKeyCount.reserve(R_LENGTH);
                    for(const auto &jt : opencl_res) openclKeyCount[jt.key]++;
                    for(const auto &jt : stdRes) stdKeyCount[jt.key]++;
                    if(openclKeyCount.size() != stdKeyCount.size()) {
                        opencl_pass = false;
                    } else {
                        for(const auto &kv : openclKeyCount) {
                            auto it = stdKeyCount.find(kv.first);
                            if(it == stdKeyCount.end() || it->second != kv.second) { opencl_pass = false; break; }
                        }
                    }
                }
                std::cout << "OpenCL Verification: " << (opencl_pass ? "PASS" : "FAIL") << "\n";
            }
        }

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

    // Verification: Compare CPU join with Standard join
    if(run_cpu_join && run_std_join) {
        std::cout << "\n=== Verification (CPU vs Standard) ===" << std::endl;
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
    }
    
    return 0;
}

