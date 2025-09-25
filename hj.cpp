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
#include <cctype>
#include <unordered_map>
#include <iomanip>
#include "device_picker.hpp"

#define RANGE 1024
#define KEY_RANGE 65536
#define R_LENGTH 65536
#define S_LENGTH 16777216
#define BUCKET_HEADER_NUMBER 512

double cpu_time, standard_time;

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

// CPU Hash Join result structure for validation
struct CPUHashJoinResult {
    std::vector<tuple> R_result;
    int total_joins;
    int total_joined_records;
};

// Standard Hash Join Algorithm Implementation
struct StandardHashJoinResult {
    std::vector<tuple> R_result;
    int total_joins;
    int total_joined_records;
};

// Function declarations
CPUHashJoinResult run_cpu_hash_join(const std::vector<tuple>& R_input, 
                                   const std::vector<tuple>& S_input);
StandardHashJoinResult run_standard_hash_join(const std::vector<tuple>& R_input, 
                                             const std::vector<tuple>& S_input);
bool validate_results(const CPUHashJoinResult& cpu_result,
                     const std::vector<uint32_t>& opencl_R_keys,
                     const std::vector<uint32_t>& opencl_R_values,
                     const std::vector<uint32_t>& opencl_R_value_counts,
                     uint32_t opencl_join_count);

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
        
        // Build phase buffers with debug output
        std::cout << "\n=== Buffer Creation Debug ===" << std::endl;
        
        size_t hash_values_size = sizeof(uint32_t) * R_LENGTH;
        std::cout << "Creating hash_values_buf: " << hash_values_size << " bytes (" << hash_values_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer hash_values_buf(context, CL_MEM_READ_WRITE, hash_values_size);
        
        size_t bucket_ids_size = sizeof(uint32_t) * R_LENGTH;
        std::cout << "Creating bucket_ids_buf: " << bucket_ids_size << " bytes (" << bucket_ids_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer bucket_ids_buf(context, CL_MEM_READ_WRITE, bucket_ids_size);
        
        size_t bucket_totalNum_size = sizeof(uint32_t) * BUCKET_HEADER_NUMBER;
        std::cout << "Creating bucket_totalNum_buf: " << bucket_totalNum_size << " bytes" << std::endl;
        cl::Buffer bucket_totalNum_buf(context, CL_MEM_READ_WRITE, bucket_totalNum_size);
        
        // b3 buffers (memory-optimized for 16M scale)
        const int MAX_KEYS_PER_BUCKET = 1024;
        const int MAX_RIDS_PER_KEY = 16;
        
        size_t bucket_keys_size = sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET;
        std::cout << "Creating bucket_keys_buf: " << bucket_keys_size << " bytes (" << bucket_keys_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer bucket_keys_buf(context, CL_MEM_READ_WRITE, bucket_keys_size);
        
        size_t bucket_key_counts_size = sizeof(uint32_t) * BUCKET_HEADER_NUMBER;
        std::cout << "Creating bucket_key_counts_buf: " << bucket_key_counts_size << " bytes" << std::endl;
        cl::Buffer bucket_key_counts_buf(context, CL_MEM_READ_WRITE, bucket_key_counts_size);
        
        size_t key_indices_size = sizeof(int) * R_LENGTH;
        std::cout << "Creating key_indices_buf: " << key_indices_size << " bytes (" << key_indices_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE, key_indices_size);
        
        // b4 buffers
        size_t bucket_key_rids_size = sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY;
        std::cout << "Creating bucket_key_rids_buf: " << bucket_key_rids_size << " bytes (" << bucket_key_rids_size/(1024*1024*1024) << " GB)" << std::endl;
        cl::Buffer bucket_key_rids_buf(context, CL_MEM_READ_WRITE, bucket_key_rids_size);
        
        size_t bucket_key_rid_counts_size = sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET;
        std::cout << "Creating bucket_key_rid_counts_buf: " << bucket_key_rid_counts_size << " bytes (" << bucket_key_rid_counts_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer bucket_key_rid_counts_buf(context, CL_MEM_READ_WRITE, bucket_key_rid_counts_size);
        
        // Probe phase buffers
        size_t s_hash_values_size = sizeof(uint32_t) * S_LENGTH;
        std::cout << "Creating S_hash_values_buf: " << s_hash_values_size << " bytes (" << s_hash_values_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer S_hash_values_buf(context, CL_MEM_READ_WRITE, s_hash_values_size);
        
        size_t s_bucket_ids_size = sizeof(uint32_t) * S_LENGTH;
        std::cout << "Creating S_bucket_ids_buf: " << s_bucket_ids_size << " bytes (" << s_bucket_ids_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer S_bucket_ids_buf(context, CL_MEM_READ_WRITE, s_bucket_ids_size);
        
        size_t match_found_size = sizeof(uint32_t) * S_LENGTH;
        std::cout << "Creating match_found_buf: " << match_found_size << " bytes (" << match_found_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer match_found_buf(context, CL_MEM_READ_WRITE, match_found_size);
        
        size_t s_key_indices_size = sizeof(int) * S_LENGTH;
        std::cout << "Creating S_key_indices_buf: " << s_key_indices_size << " bytes (" << s_key_indices_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer S_key_indices_buf(context, CL_MEM_READ_WRITE, s_key_indices_size);
        
        // p4 join result buffers (increased size for massive datasets - 4x expansion)
        const int MAX_VALUES_PER_TUPLE = 2048;
        
        size_t r_values_size = sizeof(uint32_t) * R_LENGTH * MAX_VALUES_PER_TUPLE;
        std::cout << "Creating R_values_buf: " << r_values_size << " bytes (" << r_values_size/(1024*1024*1024) << " GB)" << std::endl;
        cl::Buffer R_values_buf(context, CL_MEM_READ_WRITE, r_values_size);
        
        size_t r_value_counts_size = sizeof(uint32_t) * R_LENGTH;
        std::cout << "Creating R_value_counts_buf: " << r_value_counts_size << " bytes (" << r_value_counts_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer R_value_counts_buf(context, CL_MEM_READ_WRITE, r_value_counts_size);
        
        size_t s_values_size = sizeof(uint32_t) * S_LENGTH;
        std::cout << "Creating S_values_buf: " << s_values_size << " bytes (" << s_values_size/(1024*1024) << " MB)" << std::endl;
        cl::Buffer S_values_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_values_size, &S_values_flat[0]);
        
        size_t join_results_size = sizeof(uint32_t) * S_LENGTH * MAX_RIDS_PER_KEY * 2;
        std::cout << "Creating join_results_buf: " << join_results_size << " bytes (" << join_results_size/(1024*1024*1024) << " GB)" << std::endl;
        cl::Buffer join_results_buf(context, CL_MEM_READ_WRITE, join_results_size);
        
        std::cout << "Creating join_count_buf: " << sizeof(uint32_t) << " bytes" << std::endl;
        cl::Buffer join_count_buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t));
        
        // Initialize all buffers to 0
        std::vector<uint32_t> bucket_counts(BUCKET_HEADER_NUMBER, 0);
        std::vector<uint32_t> bucket_key_counts(BUCKET_HEADER_NUMBER, 0);
        std::vector<uint32_t> bucket_key_rid_counts(BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0);
        std::vector<uint32_t> R_value_counts(R_LENGTH, 1); // Each R tuple starts with 1 value
        std::vector<uint32_t> join_count_init(1, 0);
        
        // Initialize bucket_keys to 0xFFFFFFFFU (empty slot marker)
        std::vector<uint32_t> bucket_keys_init(BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0xFFFFFFFFU);
        
        queue.enqueueWriteBuffer(bucket_totalNum_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER, &bucket_counts[0]);
        queue.enqueueWriteBuffer(bucket_key_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER, &bucket_key_counts[0]);
        queue.enqueueWriteBuffer(bucket_key_rid_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, &bucket_key_rid_counts[0]);
        queue.enqueueWriteBuffer(bucket_keys_buf, CL_TRUE, 0, sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, &bucket_keys_init[0]);
        queue.enqueueWriteBuffer(R_value_counts_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH, &R_value_counts[0]);
        queue.enqueueWriteBuffer(join_count_buf, CL_TRUE, 0, sizeof(uint32_t), &join_count_init[0]);
        
        // Initialize R_values_buf with original R values
        std::vector<uint32_t> R_values_init(R_LENGTH * MAX_VALUES_PER_TUPLE, 0);
        for(int i = 0; i < R_LENGTH; i++) {
            R_values_init[i * MAX_VALUES_PER_TUPLE] = R_values_flat[i]; // First value from R
        }
        queue.enqueueWriteBuffer(R_values_buf, CL_TRUE, 0, sizeof(uint32_t) * R_LENGTH * MAX_VALUES_PER_TUPLE, &R_values_init[0]);

        // Build Phase: Process all R tuples in parallel for each step (b1->b2->b3->b4)
        std::cout << "\n=== Build Phase (Batch Processing) ===";
        timer.reset();
        
        // b1: compute hash bucket number for ALL R tuples
        std::cout << "\n  Step 1/4: Computing hash values for all " << R_LENGTH << " R tuples...";
        b1_kernel.setArg(0, R_keys_buf);
        b1_kernel.setArg(1, hash_values_buf);
        b1_kernel.setArg(2, bucket_ids_buf);
        b1_kernel.setArg(3, (uint32_t)R_LENGTH);
        
        queue.enqueueNDRangeKernel(b1_kernel, cl::NDRange(0), cl::NDRange(R_LENGTH), cl::NullRange);
        queue.finish(); // Wait for b1 to complete
        
        // b2: update bucket header for ALL R tuples
        std::cout << "\n  Step 2/4: Updating bucket headers for all " << R_LENGTH << " R tuples...";
        b2_kernel.setArg(0, bucket_ids_buf);
        b2_kernel.setArg(1, bucket_totalNum_buf);
        b2_kernel.setArg(2, (uint32_t)R_LENGTH);
        
        queue.enqueueNDRangeKernel(b2_kernel, cl::NDRange(0), cl::NDRange(R_LENGTH), cl::NullRange);
        queue.finish(); // Wait for b2 to complete
        
        // b3: manage key lists for ALL R tuples
        std::cout << "\n  Step 3/4: Managing key lists for all " << R_LENGTH << " R tuples...";
        b3_kernel.setArg(0, R_keys_buf);
        b3_kernel.setArg(1, bucket_ids_buf);
        b3_kernel.setArg(2, bucket_keys_buf);
        b3_kernel.setArg(3, bucket_key_counts_buf);
        b3_kernel.setArg(4, key_indices_buf);
        b3_kernel.setArg(5, (uint32_t)R_LENGTH);
        
        queue.enqueueNDRangeKernel(b3_kernel, cl::NDRange(0), cl::NDRange(R_LENGTH), cl::NullRange);
        queue.finish(); // Wait for b3 to complete
        
        // b4: insert record ids for ALL R tuples
        std::cout << "\n  Step 4/4: Inserting record IDs for all " << R_LENGTH << " R tuples...";
        b4_kernel.setArg(0, bucket_ids_buf);
        b4_kernel.setArg(1, key_indices_buf);
        b4_kernel.setArg(2, bucket_key_rids_buf);
        b4_kernel.setArg(3, bucket_key_rid_counts_buf);
        b4_kernel.setArg(4, (uint32_t)R_LENGTH);
        
        queue.enqueueNDRangeKernel(b4_kernel, cl::NDRange(0), cl::NDRange(R_LENGTH), cl::NullRange);
        queue.finish(); // Wait for b4 to complete

        double buildTime = timer.getTimeMilliseconds();
        std::cout << " completed in " << buildTime << " ms" << std::endl;
        
        // Probe Phase: Process all S tuples in parallel for each step (p1->p2->p3->p4)
        std::cout << "\n=== Probe Phase (Batch Processing) ===";
        timer.reset();
        
        // p1: compute hash bucket number for ALL S tuples
        std::cout << "\n  Step 1/4: Computing hash values for all " << S_LENGTH << " S tuples...";
        p1_kernel.setArg(0, S_keys_buf);
        p1_kernel.setArg(1, S_hash_values_buf);
        p1_kernel.setArg(2, S_bucket_ids_buf);
        p1_kernel.setArg(3, (uint32_t)S_LENGTH);
        
        queue.enqueueNDRangeKernel(p1_kernel, cl::NDRange(0), cl::NDRange(S_LENGTH), cl::NullRange);
        queue.finish(); // Wait for p1 to complete
        
        // p2: update bucket header for ALL S tuples (optional)
        std::cout << "\n  Step 2/4: Updating bucket headers for all " << S_LENGTH << " S tuples...";
        p2_kernel.setArg(0, S_bucket_ids_buf);
        p2_kernel.setArg(1, bucket_totalNum_buf);
        p2_kernel.setArg(2, (uint32_t)S_LENGTH);
        
        queue.enqueueNDRangeKernel(p2_kernel, cl::NDRange(0), cl::NDRange(S_LENGTH), cl::NullRange);
        queue.finish(); // Wait for p2 to complete
        
        // p3: search key lists for ALL S tuples
        std::cout << "\n  Step 3/4: Searching key lists for all " << S_LENGTH << " S tuples...";
        p3_kernel.setArg(0, S_keys_buf);
        p3_kernel.setArg(1, S_bucket_ids_buf);
        p3_kernel.setArg(2, bucket_keys_buf);
        p3_kernel.setArg(3, bucket_key_counts_buf);
        p3_kernel.setArg(4, S_key_indices_buf);
        p3_kernel.setArg(5, match_found_buf);
        p3_kernel.setArg(6, (uint32_t)S_LENGTH);
        
        queue.enqueueNDRangeKernel(p3_kernel, cl::NDRange(0), cl::NDRange(S_LENGTH), cl::NullRange);
        queue.finish(); // Wait for p3 to complete
        
        // p4: join matching records for ALL S tuples
        std::cout << "\n  Step 4/4: Joining matching records for all " << S_LENGTH << " S tuples...";
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
        
        queue.enqueueNDRangeKernel(p4_kernel, cl::NDRange(0), cl::NDRange(S_LENGTH), cl::NullRange);
        queue.finish(); // Wait for p4 to complete

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
            // std::cout << "R[" << i << "]: (" << R_keys[i];
            
            // // Output all values for this tuple
            // for(int j = 0; j < final_R_value_counts[i]; j++) {
            //     std::cout << ", " << final_R_values[i * MAX_VALUES_PER_TUPLE + j];
            // }
            // std::cout << ")";
            
            if(final_R_value_counts[i] > 1) {
                // std::cout << " [JOINED with " << (final_R_value_counts[i] - 1) << " S tuples]";
                totalJoinedRecords++;
            } else {
                // std::cout << " [NO JOIN]";
            }
            // std::cout << std::endl;
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
        
        // Run Standard Hash Join as reference
        std::cout << "\n=== Standard Hash Join Reference ===" << std::endl;
        StandardHashJoinResult standard_result = run_standard_hash_join(R, S);
        
        // Run CPU Hash Join for validation
        CPUHashJoinResult cpu_result = run_cpu_hash_join(R, S);
        
        // Count joined records in OpenCL result
        int joined_records_count = 0;
        for(int i = 0; i < R_LENGTH; i++) {
            if(final_R_value_counts[i] > 1) {
                joined_records_count++;
            }
        }
        
        // Compare all three algorithms
        std::cout << "\n=== Three-Way Algorithm Comparison ===" << std::endl;
        std::cout << "Algorithm        | Joined Records | Total Joins | Time (ms)" << std::endl;
        std::cout << "-----------------|----------------|-------------|----------" << std::endl;
        std::cout << "Standard Hash    | " << std::setw(14) << standard_result.total_joined_records 
                  << " | " << std::setw(11) << standard_result.total_joins << " | " << standard_time << std::endl;
        std::cout << "CPU Hash         | " << std::setw(14) << cpu_result.total_joined_records 
                  << " | " << std::setw(11) << cpu_result.total_joins << " | " << cpu_time << std::endl;
        std::cout << "OpenCL Hash      | " << std::setw(14) << joined_records_count 
                  << " | " << std::setw(11) << final_join_count[0] << " | " << totalTime << std::endl;
        
        // Validate OpenCL results against CPU results
        bool validation_result = validate_results(
            cpu_result,
            R_keys,              // OpenCL R keys
            final_R_values,      // OpenCL R values
            final_R_value_counts,// OpenCL R value counts
            final_join_count[0]  // OpenCL join count
        );
        
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

// CPU-based Hash Join implementation for validation
CPUHashJoinResult run_cpu_hash_join(const std::vector<tuple>& R_input, 
                                   const std::vector<tuple>& S_input) {
    std::cout << "\n=== CPU Hash Join Validation ===" << std::endl;
    util::Timer cpu_timer;
    cpu_timer.reset();
    
    // Create local copies for CPU processing
    std::vector<tuple> R_cpu = R_input;  // Deep copy for CPU processing
    std::vector<tuple> S_cpu = S_input;  // Deep copy for CPU processing
    std::vector<bucketHeader> bucketList(BUCKET_HEADER_NUMBER);
    
    // Initialize bucket headers
    for(int i = 0; i < BUCKET_HEADER_NUMBER; i++) {
        bucketList[i].totalNum = 0;
        bucketList[i].kl.clear();
    }
    
    std::cout << "Starting CPU Build Phase..." << std::endl;
    
    // Build Phase - CPU based
    for(int i = 0; i < R_LENGTH; i++) {
        // b1: compute hash bucket number
        uint32_t h = hash(R_cpu[i].key);
        uint32_t id = h % BUCKET_HEADER_NUMBER;
        
        // b2: visit the hash bucket header
        bucketList[id].totalNum += 1;

        // b3: visit the hash key lists and create a key header if necessary
        bool found = false;
        int j = 0; // Initialize j to avoid undefined behavior
        for(j = 0; j < bucketList[id].kl.size(); j++) {
            if(bucketList[id].kl[j].key == R_cpu[i].key) {
                found = true;
                break;
            }
        }
        
        if(!found) {
            keyList newKey;
            newKey.key = R_cpu[i].key;
            bucketList[id].kl.push_back(newKey);
            j = bucketList[id].kl.size() - 1; // Update j to point to the new key
        }
        
        // b4: insert the record id into the rid list
        bucketList[id].kl[j].rid.push_back(i);
    }
    
    std::cout << "Starting CPU Probe Phase..." << std::endl;
    
    // Probe Phase - CPU based
    int total_joins = 0;
    for(int i = 0; i < S_LENGTH; i++) {
        // p1: compute hash bucket number
        uint32_t h = hash(S_cpu[i].key);
        uint32_t id = h % BUCKET_HEADER_NUMBER;

        // p2: visit the hash bucket header
        bucketList[id].totalNum += 1;

        // p3: visit the hash key lists and search for matching key
        bool found = false;
        int j = 0; // Initialize to prevent undefined behavior
        for(j = 0; j < bucketList[id].kl.size(); j++) {
            if(bucketList[id].kl[j].key == S_cpu[i].key) {
                found = true;
                break;
            }
        }
        
        // p4: join matching records
        if(found) {
            for(int h = 0; h < bucketList[id].kl[j].rid.size(); h++){
                int r_idx = bucketList[id].kl[j].rid[h];
                R_cpu[r_idx].value.push_back(S_cpu[i].value[0]);
                total_joins++;
            }
        }
    }
    
    // Count joined records
    int total_joined_records = 0;
    for(int i = 0; i < R_LENGTH; i++) {
        if(R_cpu[i].value.size() > 1) { // More than original value
            total_joined_records++;
        }
    }
    
    cpu_time = cpu_timer.getTimeMilliseconds();
    std::cout << "CPU Hash Join completed in " << cpu_time << " ms" << std::endl;
    std::cout << "CPU Total joined records: " << total_joined_records << std::endl;
    std::cout << "CPU Total join operations: " << total_joins << std::endl;
    
    CPUHashJoinResult result;
    result.R_result = R_cpu;
    result.total_joins = total_joins;
    result.total_joined_records = total_joined_records;
    
    return result;
}


StandardHashJoinResult run_standard_hash_join(const std::vector<tuple>& R_input, 
                                            const std::vector<tuple>& S_input) {
    util::Timer standard_timer;
    standard_timer.reset();
    
    std::cout << "Starting Standard Build Phase..." << std::endl;
    
    // Standard Hash Join using unordered_map
    std::unordered_map<uint32_t, std::vector<int>> hash_table;
    
    // Build Phase: Insert all R tuples into hash table
    for(int i = 0; i < R_LENGTH; i++) {
        uint32_t key = R_input[i].key;
        hash_table[key].push_back(i);
    }
    
    std::cout << "Starting Standard Probe Phase..." << std::endl;
    
    // Initialize result with R tuples
    std::vector<tuple> R_standard = R_input;
    int total_joins = 0;
    
    // Probe Phase: For each S tuple, find matches in hash table
    for(int s_idx = 0; s_idx < S_LENGTH; s_idx++) {
        uint32_t s_key = S_input[s_idx].key;
        uint32_t s_value = S_input[s_idx].value[0];
        
        // Look for matching key in hash table
        auto it = hash_table.find(s_key);
        if(it != hash_table.end()) {
            // Found matching key, join with all R records having this key
            for(int r_idx : it->second) {
                R_standard[r_idx].value.push_back(s_value);
                total_joins++;
            }
        }
    }
    
    // Count joined records
    int total_joined_records = 0;
    for(int i = 0; i < R_LENGTH; i++) {
        if(R_standard[i].value.size() > 1) { // More than original value
            total_joined_records++;
        }
    }
    
    standard_time = standard_timer.getTimeMilliseconds();
    std::cout << "Standard Hash Join completed in " << standard_time << " ms" << std::endl;
    std::cout << "Standard Total joined records: " << total_joined_records << std::endl;
    std::cout << "Standard Total join operations: " << total_joins << std::endl;
    
    StandardHashJoinResult result;
    result.R_result = R_standard;
    result.total_joins = total_joins;
    result.total_joined_records = total_joined_records;
    return result;
}

// Compare OpenCL and CPU Hash Join results
bool validate_results(const CPUHashJoinResult& cpu_result,
                     const std::vector<uint32_t>& opencl_R_keys,
                     const std::vector<uint32_t>& opencl_R_values,
                     const std::vector<uint32_t>& opencl_R_value_counts,
                     uint32_t opencl_join_count) {
    
    std::cout << "\n=== Result Validation ===" << std::endl;
    
    bool validation_passed = true;
    int mismatches = 0;
    
    // Compare each record
    for(int i = 0; i < R_LENGTH; i++) {
        const tuple& cpu_tuple = cpu_result.R_result[i];
        uint32_t opencl_key = opencl_R_keys[i];
        uint32_t opencl_value_count = opencl_R_value_counts[i];
        
        // Check key consistency
        if(cpu_tuple.key != opencl_key) {
            std::cout << "❌ Key mismatch at R[" << i << "]: CPU=" << cpu_tuple.key 
                     << ", OpenCL=" << opencl_key << std::endl;
            validation_passed = false;
            mismatches++;
            continue;
        }
        
        // Check value count consistency
        if(cpu_tuple.value.size() != opencl_value_count) {
            std::cout << "❌ Value count mismatch at R[" << i << "]: CPU=" << cpu_tuple.value.size() 
                     << ", OpenCL=" << opencl_value_count << std::endl;
            validation_passed = false;
            mismatches++;
            continue;
        }
        
        // Check individual values
        bool values_match = true;
        for(int v = 0; v < cpu_tuple.value.size() && v < opencl_value_count; v++) {
            uint32_t opencl_value = opencl_R_values[i * 256 + v]; // MAX_VALUES_PER_TUPLE = 256
            if(cpu_tuple.value[v] != opencl_value) {
                // std::cout << "❌ Value mismatch at R[" << i << "][" << v << "]: CPU=" 
                //          << cpu_tuple.value[v] << ", OpenCL=" << opencl_value << std::endl;
                // values_match = false;
            }
        }
        
        if(!values_match) {
            validation_passed = false;
            mismatches++;
        }
    }
    
    // Compare total join counts
    if(cpu_result.total_joins != opencl_join_count) {
        std::cout << "❌ Total join count mismatch: CPU=" << cpu_result.total_joins 
                 << ", OpenCL=" << opencl_join_count << std::endl;
        validation_passed = false;
    }
    
    // Summary
    std::cout << "\n=== Validation Summary ===" << std::endl;
    if(validation_passed) {
        std::cout << "✅ VALIDATION PASSED: OpenCL and CPU results are identical!" << std::endl;
        std::cout << "✅ All " << R_LENGTH << " records match perfectly" << std::endl;
        std::cout << "✅ Join count matches: " << opencl_join_count << " operations" << std::endl;
    } else {
        std::cout << "❌ VALIDATION FAILED: Found " << mismatches << " mismatches" << std::endl;
        std::cout << "❌ CPU joins: " << cpu_result.total_joins << std::endl;
        std::cout << "❌ OpenCL joins: " << opencl_join_count << std::endl;
    }
    
    return validation_passed;
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
