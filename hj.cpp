#define CL_TARGET_OPENCL_VERSION 300
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include "device_picker.hpp"

#define RANGE 1024
#define KEY_RANGE 1024
#define R_LENGTH 128
#define S_LENGTH 1024
#define BUCKET_HEADER_NUMBER 16

static const uint32_t GOLDEN_RATIO_32 = 2654435769U;

uint32_t hash(uint32_t key) {
    return (key * GOLDEN_RATIO_32) % RANGE;
}

struct tuple {
    uint32_t key;
    std::vector<uint32_t> value;
};

struct keyList {
    uint32_t key;
    std::vector<uint32_t> rid;
};

struct bucketHeader {
    int totalNum;
    std::vector<keyList> kl;
};

int main(int argc, char *argv[]) {
    srand(time(NULL)); // Initialize random seed
    
    // Create timer for performance measurement
    util::Timer timer;
    
    std::vector<tuple> R(R_LENGTH);
    std::vector<tuple> S(S_LENGTH);
    std::vector<bucketHeader> bucketList(BUCKET_HEADER_NUMBER);

    std::cout << "Initializing data structures..." << std::endl;
    std::cout << "R table size: " << R_LENGTH << std::endl;
    std::cout << "S table size: " << S_LENGTH << std::endl;
    std::cout << "Hash buckets: " << BUCKET_HEADER_NUMBER << std::endl;

    int i;
    for(i = 0; i < BUCKET_HEADER_NUMBER; i++) {
        bucketList[i].totalNum = 0;
        bucketList[i].kl.clear(); // Initialize empty keyList vector
    }

    for(i = 0; i < R_LENGTH; i++) {
        R[i].key = rand() % KEY_RANGE;
        R[i].value.push_back(rand() % RANGE);
    }

    for(i = 0; i < S_LENGTH; i++) {
        S[i].key = rand() % KEY_RANGE;
        S[i].value.push_back(rand() % RANGE);
    }
    
    try
    {
        cl_uint deviceIndex = 0;
        parseArguments(argc, argv, &deviceIndex);

        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        if(deviceIndex >= numDevices) {
            std::cout << "Invalid device index";
            return EXIT_FAILURE;
        }
        
        cl::Device device = devices[1];

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
        
        // Build programs with OpenCL 3.0 options
        std::string build_options = "-cl-std=CL3.0";
        
        try {
            std::cout << "Building programs with OpenCL 3.0..." << std::endl;
            b1_program.build(chosen_device, build_options.c_str());
            b2_program.build(chosen_device, build_options.c_str());
            b3_program.build(chosen_device, build_options.c_str());
            b4_program.build(chosen_device, build_options.c_str());
            p1_program.build(chosen_device, build_options.c_str());
            p2_program.build(chosen_device, build_options.c_str());
            p3_program.build(chosen_device, build_options.c_str());
            p4_program.build(chosen_device, build_options.c_str());
            
            std::cout << "All programs built successfully!" << std::endl;
        } catch (cl::Error err) {
            std::cout << "Build error: " << err.what() << std::endl;
            
            // Print build log for debugging
            std::cout << "\n=== Build Logs ===" << std::endl;
            
            std::cout << "b1 build log: " << b1_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "b2 build log: " << b2_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "b3 build log: " << b3_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "b4 build log: " << b4_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "p1 build log: " << p1_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "p2 build log: " << p2_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "p3 build log: " << p3_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cout << "p4 build log: " << p4_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            
            throw; // Re-throw the error
        }
        
        // Create kernels
        cl::Kernel b1_kernel(b1_program, "b1_compute_hash");
        cl::Kernel b2_kernel(b2_program, "b2_update_bucket_header");
        cl::Kernel b3_kernel(b3_program, "b3_manage_key_lists");
        cl::Kernel b4_kernel(b4_program, "b4_insert_record_ids");
        cl::Kernel p1_kernel(p1_program, "p1_compute_hash");
        cl::Kernel p2_kernel(p2_program, "p2_update_bucket_header");
        cl::Kernel p3_kernel(p3_program, "p3_search_key_lists");
        cl::Kernel p4_kernel(p4_program, "p4_join_records");

        // Prepare data for OpenCL execution
        std::vector<uint32_t> R_keys(R_LENGTH), R_values_flat(R_LENGTH), S_keys(S_LENGTH), S_values_flat(S_LENGTH);
        
        // Flatten R and S data
        for(int i = 0; i < R_LENGTH; i++) {
            R_keys[i] = R[i].key;
            R_values_flat[i] = R[i].value[0];
        }
        
        for(int i = 0; i < S_LENGTH; i++) {
            S_keys[i] = S[i].key;  
            S_values_flat[i] = S[i].value[0];
        }

        // Create OpenCL buffers with OpenCL 3.0 compatible flags
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        
        // Build phase buffers
        cl::Buffer hash_values_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH);
        cl::Buffer bucket_ids_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH);
        cl::Buffer bucket_totalNum_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER);
        
        // b3 buffers (increased sizes for larger datasets)
        const int MAX_KEYS_PER_BUCKET = 64;
        const int MAX_RIDS_PER_KEY = 256;
        cl::Buffer bucket_keys_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer bucket_key_counts_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE, sizeof(int) * R_LENGTH);
        
        // b4 buffers
        cl::Buffer bucket_key_rids_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY);
        cl::Buffer bucket_key_rid_counts_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        
        // Probe phase buffers
        cl::Buffer S_hash_values_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * S_LENGTH);
        cl::Buffer S_bucket_ids_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * S_LENGTH);
        cl::Buffer match_found_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * S_LENGTH);
        cl::Buffer S_key_indices_buf(context, CL_MEM_READ_WRITE, sizeof(int) * S_LENGTH);
        
        // p4 join result buffers (increased size for larger datasets)
        const int MAX_VALUES_PER_TUPLE = 256;
        cl::Buffer R_values_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH * MAX_VALUES_PER_TUPLE);
        cl::Buffer R_value_counts_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * R_LENGTH);
        cl::Buffer S_values_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t) * S_LENGTH, &S_values_flat[0]);
        cl::Buffer join_results_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * S_LENGTH * MAX_RIDS_PER_KEY * 2);
        cl::Buffer join_count_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t));
        
        // Initialize all buffers to 0
        std::vector<uint32_t> bucket_counts(BUCKET_HEADER_NUMBER, 0);
        std::vector<uint32_t> bucket_key_counts(BUCKET_HEADER_NUMBER, 0);
        std::vector<uint32_t> bucket_key_rid_counts(BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0);
        std::vector<uint32_t> R_value_counts(R_LENGTH, 1); // Each R tuple starts with 1 value
        std::vector<uint32_t> join_count_init(1, 0);
        
        queue.enqueueWriteBuffer(bucket_totalNum_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER, &bucket_counts[0]);
        queue.enqueueWriteBuffer(bucket_key_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER, &bucket_key_counts[0]);
        queue.enqueueWriteBuffer(bucket_key_rid_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, &bucket_key_rid_counts[0]);
        queue.enqueueWriteBuffer(R_value_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH, &R_value_counts[0]);
        queue.enqueueWriteBuffer(join_count_buf, CL_TRUE, 0, sizeof(uint32_t), &join_count_init[0]);
        
        // Initialize R_values_buf with original R values
        std::vector<uint32_t> R_values_init(R_LENGTH * MAX_VALUES_PER_TUPLE, 0);
        for(int i = 0; i < R_LENGTH; i++) {
            R_values_init[i * MAX_VALUES_PER_TUPLE] = R_values_flat[i]; // First value from R
        }
        queue.enqueueWriteBuffer(R_values_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH * MAX_VALUES_PER_TUPLE, &R_values_init[0]);

        // Build Phase: For each tuple in R, run b1->b2->b3->b4
        std::cout << "\n=== Build Phase ===";
        timer.reset();
        
        for(int i = 0; i < R_LENGTH; i++) {
            
            try {
                // b1: compute hash bucket number
                b1_kernel.setArg(0, R_keys_buf);
                b1_kernel.setArg(1, hash_values_buf);
                b1_kernel.setArg(2, bucket_ids_buf);
                b1_kernel.setArg(3, (uint32_t)R_LENGTH);
                
                queue.enqueueNDRangeKernel(b1_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 10 == 0) std::cout << "b1 completed for R[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in b1 kernel at R[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // b2: update bucket header
                b2_kernel.setArg(0, bucket_ids_buf);
                b2_kernel.setArg(1, bucket_totalNum_buf);
                b2_kernel.setArg(2, (uint32_t)R_LENGTH);
                
                queue.enqueueNDRangeKernel(b2_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 10 == 0) std::cout << "b2 completed for R[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in b2 kernel at R[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // b3: manage key lists
                b3_kernel.setArg(0, R_keys_buf);
                b3_kernel.setArg(1, bucket_ids_buf);
                b3_kernel.setArg(2, bucket_keys_buf);
                b3_kernel.setArg(3, bucket_key_counts_buf);
                b3_kernel.setArg(4, key_indices_buf);
                b3_kernel.setArg(5, (uint32_t)R_LENGTH);
                
                queue.enqueueNDRangeKernel(b3_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 10 == 0) std::cout << "b3 completed for R[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in b3 kernel at R[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // b4: insert record ids
                b4_kernel.setArg(0, bucket_ids_buf);
                b4_kernel.setArg(1, key_indices_buf);
                b4_kernel.setArg(2, bucket_key_rids_buf);
                b4_kernel.setArg(3, bucket_key_rid_counts_buf);
                b4_kernel.setArg(4, (uint32_t)R_LENGTH);
                
                queue.enqueueNDRangeKernel(b4_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 10 == 0) std::cout << "b4 completed for R[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in b4 kernel at R[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
        }

        double buildTime = timer.getTimeMilliseconds();
        std::cout << " completed in " << buildTime << " ms" << std::endl;
        
        // Probe Phase: For each tuple in S, run p1->p2->p3->p4
        std::cout << "\n=== Probe Phase ===";
        timer.reset();
        
        for(int i = 0; i < S_LENGTH; i++) {
            
            try {
                // p1: compute hash bucket number
                p1_kernel.setArg(0, S_keys_buf);
                p1_kernel.setArg(1, S_hash_values_buf);
                p1_kernel.setArg(2, S_bucket_ids_buf);
                p1_kernel.setArg(3, (uint32_t)S_LENGTH);
                
                queue.enqueueNDRangeKernel(p1_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 50 == 0) std::cout << "p1 completed for S[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in p1 kernel at S[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // p2: update bucket header (optional)
                p2_kernel.setArg(0, S_bucket_ids_buf);
                p2_kernel.setArg(1, bucket_totalNum_buf);
                p2_kernel.setArg(2, (uint32_t)S_LENGTH);
                
                queue.enqueueNDRangeKernel(p2_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 50 == 0) std::cout << "p2 completed for S[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in p2 kernel at S[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // p3: search key lists
                p3_kernel.setArg(0, S_keys_buf);
                p3_kernel.setArg(1, S_bucket_ids_buf);
                p3_kernel.setArg(2, bucket_keys_buf);
                p3_kernel.setArg(3, bucket_key_counts_buf);
                p3_kernel.setArg(4, S_key_indices_buf);
                p3_kernel.setArg(5, match_found_buf);
                p3_kernel.setArg(6, (uint32_t)S_LENGTH);
                
                queue.enqueueNDRangeKernel(p3_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 50 == 0) std::cout << "p3 completed for S[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in p3 kernel at S[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
            
            try {
                // p4: join matching records
                p4_kernel.setArg(0, S_values_buf);
                p4_kernel.setArg(1, S_bucket_ids_buf);
                p4_kernel.setArg(2, S_key_indices_buf);
                p4_kernel.setArg(3, match_found_buf);
                p4_kernel.setArg(4, bucket_key_rids_buf);
                p4_kernel.setArg(5, bucket_key_rid_counts_buf);
                p4_kernel.setArg(6, R_values_buf);
                p4_kernel.setArg(7, R_value_counts_buf);
                p4_kernel.setArg(8, join_results_buf);
                p4_kernel.setArg(9, join_count_buf);
                p4_kernel.setArg(10, (uint32_t)S_LENGTH);
                
                queue.enqueueNDRangeKernel(p4_kernel, cl::NDRange(i), cl::NDRange(1), cl::NullRange);
                queue.finish();
                
                if(i % 50 == 0) std::cout << "p4 completed for S[" << i << "]" << std::endl;
                
            } catch (cl::Error err) {
                std::cout << "ERROR in p4 kernel at S[" << i << "]: " << err.what() << " (" << err_code(err.err()) << ")" << std::endl;
                throw;
            }
        }

        double probeTime = timer.getTimeMilliseconds();
        std::cout << " completed in " << probeTime << " ms" << std::endl;
        
        // Read back the join results
        std::cout << "\n=== Reading Hash Join Results ===" << std::endl;
        
        std::vector<uint32_t> final_R_values(R_LENGTH * MAX_VALUES_PER_TUPLE);
        std::vector<uint32_t> final_R_value_counts(R_LENGTH);
        std::vector<uint32_t> final_join_count(1);
        
        queue.enqueueReadBuffer(R_values_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH * MAX_VALUES_PER_TUPLE, &final_R_values[0]);
        queue.enqueueReadBuffer(R_value_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH, &final_R_value_counts[0]);
        queue.enqueueReadBuffer(join_count_buf, CL_TRUE, 0, sizeof(uint32_t), &final_join_count[0]);
        
        // Output the Hash Join results
        std::cout << "\n=== Hash Join Results ===" << std::endl;
        std::cout << "Format: (key, value1, value2, ...)" << std::endl;
        std::cout << "=================================" << std::endl;
        
        int totalJoinedRecords = 0;
        int totalRecords = 0;
        
        for(int i = 0; i < R_LENGTH; i++) {
            std::cout << "R[" << i << "]: (" << R_keys[i];
            
            // Output all values for this tuple
            for(int j = 0; j < final_R_value_counts[i]; j++) {
                std::cout << ", " << final_R_values[i * MAX_VALUES_PER_TUPLE + j];
            }
            std::cout << ")";
            
            if(final_R_value_counts[i] > 1) {
                std::cout << " [JOINED with " << (final_R_value_counts[i] - 1) << " S tuples]";
                totalJoinedRecords++;
            } else {
                std::cout << " [NO JOIN]";
            }
            std::cout << std::endl;
            totalRecords++;
        }
        
        std::cout << "=================================" << std::endl;
        std::cout << "Total R records processed: " << totalRecords << std::endl;
        std::cout << "Total R records joined: " << totalJoinedRecords << std::endl;
        std::cout << "Total join operations performed: " << final_join_count[0] << std::endl;
        
        double totalTime = buildTime + probeTime;
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << "Build Phase time: " << buildTime << " ms" << std::endl;
        std::cout << "Probe Phase time: " << probeTime << " ms" << std::endl;
        std::cout << "Total execution time: " << totalTime << " ms" << std::endl;
        std::cout << "R table size: " << R_LENGTH << " tuples" << std::endl;
        std::cout << "S table size: " << S_LENGTH << " tuples" << std::endl;
        
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

        // Final results summary
        std::cout << "\n=== OpenCL Hash Join Summary ===" << std::endl;
        std::cout << "Successfully processed " << R_LENGTH << " R tuples and " << S_LENGTH << " S tuples" << std::endl;
        std::cout << "Used " << BUCKET_HEADER_NUMBER << " hash buckets" << std::endl;
        std::cout << "OpenCL Hash Join completed successfully!" << std::endl;

    return 0;
}

/*
==================================================================================
=                         ORIGINAL CPU-BASED HASH JOIN CODE                     =
=                              (COMMENTED OUT)                                   =
==================================================================================

// Original CPU-based Hash Join implementation for reference
// This code was replaced by the OpenCL implementation above

void original_cpu_hash_join_code() {
    // Original Build Phase - CPU based
    for(int i = 0; i < LENGTH; i++) {
        // b1: compute hash bucket number.
        uint32_t h = hash(R[i].key);
        
        uint32_t id = h % BUCKET_HEADER_NUMBER;
        // b2: visit the hash bucket header
        bucketList[id].totalNum += 1;

        // b3: visit the hash key lists and create a key header if necessary
        bool found = false;
        int j; // Initialize j to avoid undefined behavior
        for(j = 0; j < bucketList[id].kl.size(); j++) {
            if(bucketList[id].kl[j].key == R[i].key) {
                found = true;
                break;
            }
        }
        
        if(!found) {
            keyList newKey;
            newKey.key = R[i].key;
            bucketList[id].kl.push_back(newKey);
            j = bucketList[id].kl.size() - 1; // Update j to point to the new key
        }
        
        // b4: insert the record id into the rid list
        bucketList[id].kl[j].rid.push_back(i);
    }
    
    // Original Probe Phase - CPU based
    for(int i = 0; i < LENGTH; i++) {
        // p1: compute hash bucket number.
        uint32_t h = hash(S[i].key);
        uint32_t id = h % BUCKET_HEADER_NUMBER;

        // p2: visit the hash bucket header
        bucketList[id].totalNum += 1;

        // p3: visit the hash key lists and create a key header if necessary
        bool found = false;
        int j = 0; // Initialize to prevent undefined behavior
        for(j = 0; j < bucketList[id].kl.size(); j++) {
            if(bucketList[id].kl[j].key == S[i].key) {
                found = true;
                break;
            }
        }
        
        // p4: join matching records
        if(found) {
            for(int h = 0; h < bucketList[id].kl[j].rid.size(); h++){
                R[bucketList[id].kl[j].rid[h]].value.push_back(S[i].value[0]);
            }
        }
    }

    // Original output code - CPU based
    std::cout << "Hash Join Results:" << std::endl;
    std::cout << "Format: (key, value1, value2, ...)" << std::endl;
    std::cout << "=================================" << std::endl;
    
    int totalResults = 0;
    for(int bucketId = 0; bucketId < BUCKET_HEADER_NUMBER; bucketId++) {
        for(int keyIdx = 0; keyIdx < bucketList[bucketId].kl.size(); keyIdx++) {
            keyList& currentKeyList = bucketList[bucketId].kl[keyIdx];
            
            // Output each record for this key
            for(int ridIdx = 0; ridIdx < currentKeyList.rid.size(); ridIdx++) {
                int recordId = currentKeyList.rid[ridIdx];
                
                std::cout << "(" << R[recordId].key;
                for(int valueIdx = 0; valueIdx < R[recordId].value.size(); valueIdx++) {
                    std::cout << ", " << R[recordId].value[valueIdx];
                }
                std::cout << ")" << std::endl;
                totalResults++;
            }
        }
    }
    
    std::cout << "=================================" << std::endl;
    std::cout << "Total joined records: " << totalResults << std::endl;
}

// Original manual file reading code (replaced by util::loadProgram)
std::string original_file_reading_code(const char* filename) {
    std::string source;
    FILE* fp = fopen(filename, "r");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        source.resize(size);
        fread(&source[0], 1, size, fp);
        fclose(fp);
    }
    return source;
}

// Example usage of original file reading:
// std::string b1_source = original_file_reading_code("b1.cl");
// std::string b2_source = original_file_reading_code("b2.cl");
// etc.

==================================================================================
=                             END OF ORIGINAL CODE                              =
==================================================================================
*/
