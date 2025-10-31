#include "hj.hpp"
#include "datagen.cpp"
#include "param.hpp"
#include "util.hpp"

#include "cl.hpp"
#include "device_picker.hpp"
#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <unordered_map>
#include <vector>

static std::vector<JoinedTuple>
run_standard_hash_join(const std::vector<Tuple> &R,
                       const std::vector<Tuple> &S) {
  // Build hash table from R: key -> list of R rids
  std::unordered_map<uint32_t, std::vector<uint32_t>> rIndex;
  rIndex.reserve(static_cast<size_t>(R_LENGTH) * 2);
  for (const auto &t : R) {
    rIndex[t.key].push_back(t.rid);
  }

  // Probe with S and emit joins
  std::vector<JoinedTuple> out;
  // Rough reservation heuristic to reduce reallocations
  out.reserve(static_cast<size_t>(S_LENGTH) / 4);
  for (const auto &s : S) {
    auto it = rIndex.find(s.key);
    if (it == rIndex.end())
      continue;
    const std::vector<uint32_t> &rids = it->second;
    for (size_t i = 0; i < rids.size(); i++) {
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
  bool run_bench = false;

  for (int arg_i = 1; arg_i < argc; arg_i++) {
    if (strcmp(argv[arg_i], "--cpu") == 0) {
      run_cpu_join = true;
    } else if (strcmp(argv[arg_i], "--std") == 0) {
      run_std_join = true;
    } else if (strcmp(argv[arg_i], "--bench") == 0) {
      run_bench = true;
    } else if (strcmp(argv[arg_i], "--help") == 0 ||
               strcmp(argv[arg_i], "-h") == 0) {
      std::cout
          << "Usage: " << argv[0] << " [device_index] [options]\n"
          << "Options:\n"
          << "  --cpu     Run CPU hash join\n"
          << "  --std     Run standard hash join\n"
          << "  --bench   Benchmark to find optimal WORK_RATIO_GPU\n"
          << "  --help, -h     Show this help message\n"
          << "\nExample:\n"
          << "  " << argv[0]
          << " 0              # Run all joins on GPU device 0\n"
          << "  " << argv[0]
          << " 0 --skip-cpu   # Run only standard and OpenCL joins\n"
          << "  " << argv[0]
          << " 1 --skip-std   # Run only CPU and OpenCL joins on CPU device\n";
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
  if (run_cpu_join) {
    std::cout << "=== CPU Hash Join ===\n";
    timer.reset();
    for (int i = 0; i < R_LENGTH; i++) {
      // b1: compute hash bucket number
      Tuple &tmpTuple = R[i];
      uint32_t id = hash(tmpTuple.key);
      // b2: visit the hash bucket header
      BucketHeader &tmpHeader = bucketList[id];
      // b3: visit the hash key lists and create a key header if necessary
      int j = 0;
      bool found = false;
      for (j = 0; j < tmpHeader.totalNum; j++) {
        if (tmpTuple.key == tmpHeader.keyList[j].key) {
          found = true;
          break;
        }
      }
      if (!found) {
        KeyHeader newKey;
        tmpHeader.totalNum++;
        newKey.key = tmpTuple.key;
        tmpHeader.keyList.push_back(newKey);
      }

      // b4: insert the rid into the rid list
      tmpHeader.keyList[j].ridList.push_back(tmpTuple.rid);
    }
    for (int i = 0; i < S_LENGTH; i++) {
      // p1: compute hash bucket number
      Tuple &tmpTuple = S[i];
      uint32_t id = hash(tmpTuple.key);
      // p2: visit the hash bucket header
      BucketHeader &tmpHeader = bucketList[id];
      if (!tmpHeader.totalNum)
        continue;
      // p3: visit the hash key lists
      int j = 0;
      bool found = false;
      for (j = 0; j < tmpHeader.totalNum; j++) {
        if (tmpTuple.key == tmpHeader.keyList[j].key) {
          found = true;
          break;
        }
      }

      // p4: visit the matching build tuple to compare keys and produce output
      // tuple
      if (found) {
        for (int h = 0; h < tmpHeader.keyList[j].ridList.size(); h++) {
          JoinedTuple t;
          t.key = tmpTuple.key;
          t.ridR = tmpHeader.keyList[j].ridList[h];
          t.ridS = tmpTuple.rid;
          res.push_back(t);
        }
      }
    }

    std::cout << "CPU Join: " << res.size() << " tuples, "
              << timer.getTimeMilliseconds() << "ms" << std::endl;
  }

  // Run standard hash join and verify against hash-join result
  std::vector<JoinedTuple> stdRes;
  if (run_std_join) {
    std::cout << "\n=== Standard Hash Join ===" << std::endl;
    util::Timer timer2;
    timer2.reset();
    stdRes = run_standard_hash_join(R, S);
    double stdMs = timer2.getTimeMilliseconds();
    std::cout << "Standard Join: " << stdRes.size() << " tuples, " << stdMs
              << "ms\n";
  }

  // ===================== OpenCL Join ==========================

  try {
    cl_uint deviceIndex = 0;

    // Simple argument parsing: ./hj 1 or --device 1
    if (argc >= 2) {
      // Check if first argument is a number (simple format)
      if (isdigit(argv[1][0])) {
        deviceIndex = atoi(argv[1]);
        // 나머지 인자는 parseArguments에서 처리하도록 하거나
        // 여기서는 이미 deviceIndex가 설정되었으므로 넘어감
      } else {
        // Use the original parseArguments for --device format
        parseArguments(argc, argv, &deviceIndex);
      }
    }
    bool partitioned_join = false;
    // Call shj function
    std::vector<cl::Device> devices;
    unsigned numDevices = getDeviceList(devices);

    if (!partitioned_join) {
      if (deviceIndex < numDevices) {
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
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(
            program, "b3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(
            program, "b4");
        cl::make_kernel<cl::Buffer, cl::Buffer> p1(program, "p1");
        cl::make_kernel<cl::Buffer, cl::Buffer> p2(program, "p2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer>
            p3(program, "p3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer>
            p4(program, "p4");

        std::vector<uint32_t> R_keys(R_LENGTH), R_rids(R_LENGTH),
            S_keys(S_LENGTH), S_rids(S_LENGTH);
        for (int i = 0; i < R_LENGTH; i++) {
          R_keys[i] = R[i].key;
          R_rids[i] = R[i].rid;
        }
        for (int i = 0; i < S_LENGTH; i++) {
          S_keys[i] = S[i].key;
          S_rids[i] = S[i].rid;
        }

        // buffer init
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        cl::Buffer R_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_rids[0]);
        cl::Buffer S_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_rids[0]);

        // b1
        cl::Buffer R_bucket_ids_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * R_LENGTH);

        // b2
        cl::Buffer bucket_total_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * BUCKET_HEADER_NUMBER);

        // b3
        cl::Buffer bucket_keys_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * R_LENGTH);

        // b4
        cl::Buffer bucket_key_rids_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET *
                MAX_RIDS_PER_KEY);

        // p1
        cl::Buffer S_bucket_ids_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        // p2

        // p3
        cl::Buffer S_key_indices_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(int) * S_LENGTH);
        cl::Buffer S_match_found_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t) * S_LENGTH);

        // p4 - Pre-allocate large buffer: S_LENGTH * MAX_RIDS_PER_KEY
        // Each S tuple gets MAX_RIDS_PER_KEY slots - NO ATOMIC OPERATIONS
        // NEEDED
        size_t max_result_size = (size_t)S_LENGTH * MAX_RIDS_PER_KEY;
        cl::Buffer result_key_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_rid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_sid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_count_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        std::vector<uint32_t> bucket_totalNumcounts(BUCKET_HEADER_NUMBER, 0);

        // Initialize buffers to zero
        queue.enqueueWriteBuffer(bucket_total_buf, CL_TRUE, 0,
                                 sizeof(uint32_t) * BUCKET_HEADER_NUMBER,
                                 &bucket_totalNumcounts[0]);

        // Zero-initialize large buffers
        std::vector<uint32_t> bucket_key_rids_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
            0xffffffffu);
        std::vector<uint32_t> bucket_keys_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0xffffffffu);
        std::vector<uint32_t> result_count_init(S_LENGTH, 0);
        queue.enqueueWriteBuffer(bucket_key_rids_buf, CL_TRUE, 0,
                                 sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                     MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
                                 &bucket_key_rids_init[0]);
        queue.enqueueWriteBuffer(bucket_keys_buf, CL_TRUE, 0,
                                 sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                     MAX_KEYS_PER_BUCKET,
                                 &bucket_keys_init[0]);
        queue.enqueueWriteBuffer(result_count_buf, CL_TRUE, 0,
                                 sizeof(uint32_t) * S_LENGTH,
                                 &result_count_init[0]);

        // Build Phase
        std::cout << "\n=== OpenCL Build Phase ===" << std::endl;

        util::Timer opencl_timer;
        opencl_timer.reset();

        // b1: compute hash bucket number
        b1(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_keys_buf,
           R_bucket_ids_buf);

        // b2: update bucket header
        b2(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_bucket_ids_buf,
           bucket_total_buf);

        // b3: manage key lists
        b3(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_keys_buf,
           R_bucket_ids_buf, bucket_keys_buf, key_indices_buf);

        // b4: insert record ids
        b4(cl::EnqueueArgs(queue, cl::NDRange(R_LENGTH)), R_rids_buf,
           R_bucket_ids_buf, key_indices_buf, bucket_key_rids_buf);
        queue.finish();
        double build_time = opencl_timer.getTimeMilliseconds();
        std::cout << "Build Phase Total: " << build_time << " ms" << std::endl;

        // Probe Phase
        std::cout << "\n=== OpenCL Probe Phase ===" << std::endl;

        // p1: compute hash bucket number
        opencl_timer.reset();
        p1(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_keys_buf,
           S_bucket_ids_buf);

        // p2: check bucket validity
        p2(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_bucket_ids_buf,
           bucket_total_buf);

        // p3: search key lists
        p3(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_keys_buf,
           S_bucket_ids_buf, bucket_keys_buf, S_key_indices_buf,
           S_match_found_buf);

        // p4: join matching records (NO ATOMIC OPERATIONS!)
        p4(cl::EnqueueArgs(queue, cl::NDRange(S_LENGTH)), S_keys_buf,
           S_rids_buf, S_key_indices_buf, S_match_found_buf,
           bucket_key_rids_buf, S_bucket_ids_buf, result_key_buf,
           result_rid_buf, result_sid_buf, result_count_buf);
        queue.finish();
        double probe_time = opencl_timer.getTimeMilliseconds();
        std::cout << "\nProbe Phase Total: " << probe_time << " ms"
                  << std::endl;

        std::cout << "\nOpenCL Join Total: " << build_time + probe_time << " ms"
                  << std::endl;

        // Read back result counts directly from GPU
        std::vector<uint32_t> result_counts(S_LENGTH);
        queue.enqueueReadBuffer(result_count_buf, CL_TRUE, 0,
                                sizeof(uint32_t) * S_LENGTH, &result_counts[0]);

        // Calculate total number of results
        uint32_t num_results = 0;
        for (uint32_t i = 0; i < S_LENGTH; i++) {
          num_results += result_counts[i];
        }

        std::cout << "OpenCL produced " << num_results << " joined tuples"
                  << std::endl;

        std::vector<JoinedTuple> opencl_res;

        if (num_results > 0) {
          // Read the sparse result buffers
          std::vector<uint32_t> sparse_keys(max_result_size);
          std::vector<uint32_t> sparse_rids(max_result_size);
          std::vector<uint32_t> sparse_sids(max_result_size);

          queue.enqueueReadBuffer(result_key_buf, CL_TRUE, 0,
                                  sizeof(uint32_t) * max_result_size,
                                  &sparse_keys[0]);
          queue.enqueueReadBuffer(result_rid_buf, CL_TRUE, 0,
                                  sizeof(uint32_t) * max_result_size,
                                  &sparse_rids[0]);
          queue.enqueueReadBuffer(result_sid_buf, CL_TRUE, 0,
                                  sizeof(uint32_t) * max_result_size,
                                  &sparse_sids[0]);

          // Compact results: remove empty slots
          std::vector<uint32_t> result_keys;
          std::vector<uint32_t> result_rids;
          std::vector<uint32_t> result_sids;
          result_keys.reserve(num_results);
          result_rids.reserve(num_results);
          result_sids.reserve(num_results);

          for (uint32_t i = 0; i < S_LENGTH; i++) {
            uint32_t count = result_counts[i];
            if (count > 0) {
              uint32_t base_offset = i * MAX_RIDS_PER_KEY;
              for (uint32_t j = 0; j < count; j++) {
                result_keys.push_back(sparse_keys[base_offset + j]);
                result_rids.push_back(sparse_rids[base_offset + j]);
                result_sids.push_back(sparse_sids[base_offset + j]);
              }
            }
          }

          // Convert to JoinedTuple format
          opencl_res.reserve(num_results);
          for (uint32_t i = 0; i < num_results; i++) {
            JoinedTuple jt;
            jt.key = result_keys[i];
            jt.ridR = result_rids[i];
            jt.ridS = result_sids[i];
            opencl_res.push_back(jt);
          }
        }

        // Compare OpenCL result with Standard join result
        if (run_std_join && opencl_res.size() > 0) {
          bool opencl_pass = (opencl_res.size() == stdRes.size());
          if (opencl_pass) {
            std::unordered_map<uint32_t, uint64_t> openclKeyCount;
            std::unordered_map<uint32_t, uint64_t> stdKeyCount;
            openclKeyCount.reserve(R_LENGTH);
            stdKeyCount.reserve(R_LENGTH);
            for (const auto &jt : opencl_res)
              openclKeyCount[jt.key]++;
            for (const auto &jt : stdRes)
              stdKeyCount[jt.key]++;
            if (openclKeyCount.size() != stdKeyCount.size()) {
              opencl_pass = false;
            } else {
              for (const auto &kv : openclKeyCount) {
                auto it = stdKeyCount.find(kv.first);
                if (it == stdKeyCount.end() || it->second != kv.second) {
                  opencl_pass = false;
                  break;
                }
              }
            }
          }
          std::cout << "OpenCL Verification: "
                    << (opencl_pass ? "PASS" : "FAIL") << "\n";
        }
      } else if (deviceIndex == 2) { // DD optimization
        cl::Device CPU = devices[0];
        cl::Device GPU = devices[1];

        std::string name;
        getDeviceName(CPU, name);
        std::cout << "\nUsing OpenCL CPU: " << name << "\n";
        getDeviceName(GPU, name);
        std::cout << "\nUsing OpenCL GPU: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(CPU);
        chosen_device.push_back(GPU);
        cl::Context context(chosen_device);
        cl::CommandQueue cpu_queue(context, CPU, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue gpu_queue(context, GPU, CL_QUEUE_PROFILING_ENABLE);

        cl::Program program(context, util::loadProgram("hj.cl"), true);
        cl::make_kernel<cl::Buffer, cl::Buffer> b1(program, "b1");
        cl::make_kernel<cl::Buffer, cl::Buffer> b2(program, "b2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(
            program, "b3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(
            program, "b4");
        cl::make_kernel<cl::Buffer, cl::Buffer> p1(program, "p1");
        cl::make_kernel<cl::Buffer, cl::Buffer> p2(program, "p2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer>
            p3(program, "p3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer>
            p4(program, "p4");

        std::vector<uint32_t> R_keys(R_LENGTH), R_rids(R_LENGTH),
            S_keys(S_LENGTH), S_rids(S_LENGTH);
        for (int i = 0; i < R_LENGTH; i++) {
          R_keys[i] = R[i].key;
          R_rids[i] = R[i].rid;
        }
        for (int i = 0; i < S_LENGTH; i++) {
          S_keys[i] = S[i].key;
          S_rids[i] = S[i].rid;
        }

        // buffer init
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        cl::Buffer R_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_rids[0]);
        cl::Buffer S_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_rids[0]);

        // Separated hash tables: CPU and GPU each have their own hash tables
        // Build phase: R data is split between CPU and GPU according to
        // WORK_RATIO_GPU

        // R portion 계산: WORK_RATIO_GPU 비율에 따라 분할 (4096의 배수로 조정)
        size_t gpu_R_portion =
            ((R_LENGTH * WORK_RATIO_GPU / 100) / 4096) * 4096;
        size_t cpu_R_portion = ((R_LENGTH - gpu_R_portion) / 4096) * 4096;
        gpu_R_portion = R_LENGTH - cpu_R_portion;

        std::cout << "\nR data distribution: GPU " << gpu_R_portion
                  << " tuples, CPU " << cpu_R_portion << " tuples" << std::endl;

        // CPU용 R sub-buffers
        cl_buffer_region cpu_R_keys_region = {0,
                                              sizeof(uint32_t) * cpu_R_portion};
        cl_buffer_region cpu_R_rids_region = {0,
                                              sizeof(uint32_t) * cpu_R_portion};
        cl::Buffer R_keys_cpu_buf = R_keys_buf.createSubBuffer(
            CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &cpu_R_keys_region);
        cl::Buffer R_rids_cpu_buf = R_rids_buf.createSubBuffer(
            CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &cpu_R_rids_region);

        // GPU용 R sub-buffers
        cl_buffer_region gpu_R_keys_region = {sizeof(uint32_t) * cpu_R_portion,
                                              sizeof(uint32_t) * gpu_R_portion};
        cl_buffer_region gpu_R_rids_region = {sizeof(uint32_t) * cpu_R_portion,
                                              sizeof(uint32_t) * gpu_R_portion};
        cl::Buffer R_keys_gpu_buf = R_keys_buf.createSubBuffer(
            CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &gpu_R_keys_region);
        cl::Buffer R_rids_gpu_buf = R_rids_buf.createSubBuffer(
            CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &gpu_R_rids_region);

        // CPU hash table buffers
        cl::Buffer R_bucket_ids_cpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * cpu_R_portion);
        cl::Buffer bucket_total_cpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER);
        cl::Buffer bucket_keys_cpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer key_indices_cpu_buf(context, CL_MEM_READ_WRITE,
                                       sizeof(uint32_t) * cpu_R_portion);
        cl::Buffer bucket_key_rids_cpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET *
                MAX_RIDS_PER_KEY);

        // GPU hash table buffers
        cl::Buffer R_bucket_ids_gpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * gpu_R_portion);
        cl::Buffer bucket_total_gpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER);
        cl::Buffer bucket_keys_gpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer key_indices_gpu_buf(context, CL_MEM_READ_WRITE,
                                       sizeof(uint32_t) * gpu_R_portion);
        cl::Buffer bucket_key_rids_gpu_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET *
                MAX_RIDS_PER_KEY);

        // p1
        cl::Buffer S_bucket_ids_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        // p2

        // p3
        cl::Buffer S_key_indices_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(int) * S_LENGTH);
        cl::Buffer S_match_found_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t) * S_LENGTH);

        // p4 - Pre-allocate large buffer: S_LENGTH * MAX_RIDS_PER_KEY
        // Each S tuple gets MAX_RIDS_PER_KEY slots - NO ATOMIC OPERATIONS
        // NEEDED
        size_t max_result_size = (size_t)S_LENGTH * MAX_RIDS_PER_KEY;
        cl::Buffer result_key_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_rid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_sid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_count_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        std::vector<uint32_t> bucket_totalNumcounts(BUCKET_HEADER_NUMBER, 0);

        // Initialize CPU hash table buffers
        cpu_queue.enqueueWriteBuffer(bucket_total_cpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER,
                                     &bucket_totalNumcounts[0]);
        std::vector<uint32_t> bucket_key_rids_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
            0xffffffffu);
        std::vector<uint32_t> bucket_keys_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0xffffffffu);
        cpu_queue.enqueueWriteBuffer(bucket_key_rids_cpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
                                     &bucket_key_rids_init[0]);
        cpu_queue.enqueueWriteBuffer(bucket_keys_cpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET,
                                     &bucket_keys_init[0]);

        // Initialize GPU hash table buffers
        gpu_queue.enqueueWriteBuffer(bucket_total_gpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER,
                                     &bucket_totalNumcounts[0]);
        gpu_queue.enqueueWriteBuffer(bucket_key_rids_gpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
                                     &bucket_key_rids_init[0]);
        gpu_queue.enqueueWriteBuffer(bucket_keys_gpu_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET,
                                     &bucket_keys_init[0]);

        // Initialize result buffers
        std::vector<uint32_t> result_count_init(S_LENGTH, 0);
        cpu_queue.enqueueWriteBuffer(result_count_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * S_LENGTH,
                                     &result_count_init[0]);

        std::cout << "\n=== OpenCL Build Phase (CPU-only Hash Table) ==="
                  << std::endl;
        util::Timer opencl_timer;
        opencl_timer.reset();
        // Build phase: CPU builds the entire hash table
        cl::Buffer R_bucket_ids_buf(context, CL_MEM_READ_WRITE,
                                    sizeof(uint32_t) * R_LENGTH);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * R_LENGTH);

        b1(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_keys_buf,
           R_bucket_ids_buf);
        b2(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_bucket_ids_buf,
           bucket_total_cpu_buf);
        b3(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_keys_buf,
           R_bucket_ids_buf, bucket_keys_cpu_buf, key_indices_buf);
        b4(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_rids_buf,
           R_bucket_ids_buf, key_indices_buf, bucket_key_rids_cpu_buf);
        cpu_queue.finish();
        double build_time = opencl_timer.getTimeMilliseconds();
        std::cout << "Build Phase Total: " << build_time << " ms" << std::endl;

        // Probe Phase
        if (run_bench) {
          std::cout << "\n=== OpenCL Probe Phase Benchmark ===" << std::endl;
          std::cout << "Testing WORK_RATIO_GPU from 20 to 50 in steps of 2\n";
          std::cout << "Running 10 iterations per ratio...\n" << std::endl;

          double best_avg_time = 1e9;
          int best_ratio = 0;

          for (int test_ratio = 0; test_ratio <= 30; test_ratio += 2) {
            // GPU portion 계산: test_ratio 비율에 따라 계산하고 4096의 배수로
            // 조정
            size_t gpu_portion = ((S_LENGTH * test_ratio / 100) / 4096) * 4096;
            // CPU portion 계산: 나머지를 4096의 배수로 조정
            size_t cpu_portion = ((S_LENGTH - gpu_portion) / 4096) * 4096;
            // 합이 정확히 S_LENGTH가 되도록 GPU portion 재조정
            gpu_portion = S_LENGTH - cpu_portion;

            // gpu_portion이나 cpu_portion이 0이 되지 않도록 확인
            if (gpu_portion == 0 || cpu_portion == 0) {
              std::cout << "Ratio " << test_ratio
                        << "%: Skipped (invalid portion sizes)" << std::endl;
              continue;
            }

            // GPU용 sub-buffers 생성 (ratio당 한 번만 생성)
            cl_buffer_region gpu_keys_region = {0,
                                                sizeof(uint32_t) * gpu_portion};
            cl_buffer_region gpu_rids_region = {0,
                                                sizeof(uint32_t) * gpu_portion};
            cl_buffer_region gpu_result_key_region = {
                0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region gpu_result_rid_region = {
                0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region gpu_result_sid_region = {
                0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region gpu_result_count_region = {0, sizeof(uint32_t) *
                                                               gpu_portion};

            cl::Buffer S_keys_gpu_buf = S_keys_buf.createSubBuffer(
                CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_keys_region);
            cl::Buffer S_rids_gpu_buf = S_rids_buf.createSubBuffer(
                CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_rids_region);
            cl::Buffer result_key_gpu_buf = result_key_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_result_key_region);
            cl::Buffer result_rid_gpu_buf = result_rid_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_result_rid_region);
            cl::Buffer result_sid_gpu_buf = result_sid_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_result_sid_region);
            cl::Buffer result_count_gpu_buf = result_count_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &gpu_result_count_region);

            // CPU용 sub-buffers 생성
            cl_buffer_region cpu_keys_region = {sizeof(uint32_t) * gpu_portion,
                                                sizeof(uint32_t) * cpu_portion};
            cl_buffer_region cpu_rids_region = {sizeof(uint32_t) * gpu_portion,
                                                sizeof(uint32_t) * cpu_portion};
            cl_buffer_region cpu_result_key_region = {
                sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
                sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region cpu_result_rid_region = {
                sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
                sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region cpu_result_sid_region = {
                sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
                sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
            cl_buffer_region cpu_result_count_region = {
                sizeof(uint32_t) * gpu_portion, sizeof(uint32_t) * cpu_portion};

            cl::Buffer S_keys_cpu_buf = S_keys_buf.createSubBuffer(
                CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_keys_region);
            cl::Buffer S_rids_cpu_buf = S_rids_buf.createSubBuffer(
                CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_rids_region);
            cl::Buffer result_key_cpu_buf = result_key_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_result_key_region);
            cl::Buffer result_rid_cpu_buf = result_rid_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_result_rid_region);
            cl::Buffer result_sid_cpu_buf = result_sid_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_result_sid_region);
            cl::Buffer result_count_cpu_buf = result_count_buf.createSubBuffer(
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &cpu_result_count_region);

            double total_time = 0.0;
            const int num_iterations = 10;

            for (int iter = 0; iter < num_iterations; iter++) {
              // 매 iteration마다 새로 생성하는 버퍼들 (상태가 변경되므로)
              cl::Buffer S_bucket_ids_gpu_buf(context, CL_MEM_READ_WRITE,
                                              sizeof(uint32_t) * gpu_portion);
              cl::Buffer S_key_indices_gpu_buf(context, CL_MEM_READ_WRITE,
                                               sizeof(int) * gpu_portion);
              cl::Buffer S_match_found_gpu_buf(context, CL_MEM_READ_WRITE,
                                               sizeof(uint32_t) * gpu_portion);
              cl::Buffer S_bucket_ids_cpu_buf(context, CL_MEM_READ_WRITE,
                                              sizeof(uint32_t) * cpu_portion);
              cl::Buffer S_key_indices_cpu_buf(context, CL_MEM_READ_WRITE,
                                               sizeof(int) * cpu_portion);
              cl::Buffer S_match_found_cpu_buf(context, CL_MEM_READ_WRITE,
                                               sizeof(uint32_t) * cpu_portion);

              // Probe phase 실행
              util::Timer probe_timer;
              probe_timer.reset();
              cl::Event probe_events[2];

              // GPU probe phase - GPU hash table only
              p1(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
                 S_keys_gpu_buf, S_bucket_ids_gpu_buf);
              p2(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
                 S_bucket_ids_gpu_buf, bucket_total_gpu_buf);
              p3(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
                 S_keys_gpu_buf, S_bucket_ids_gpu_buf, bucket_keys_cpu_buf,
                 S_key_indices_gpu_buf, S_match_found_gpu_buf);
              probe_events[0] = p4(
                  cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
                  S_keys_gpu_buf, S_rids_gpu_buf, S_key_indices_gpu_buf,
                  S_match_found_gpu_buf, bucket_key_rids_cpu_buf,
                  S_bucket_ids_gpu_buf, result_key_gpu_buf, result_rid_gpu_buf,
                  result_sid_gpu_buf, result_count_gpu_buf);
              gpu_queue.flush();

              // CPU probe phase - CPU hash table only
              p1(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
                 S_keys_cpu_buf, S_bucket_ids_cpu_buf);
              p2(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
                 S_bucket_ids_cpu_buf, bucket_total_cpu_buf);
              p3(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
                 S_keys_cpu_buf, S_bucket_ids_cpu_buf, bucket_keys_cpu_buf,
                 S_key_indices_cpu_buf, S_match_found_cpu_buf);
              probe_events[1] = p4(
                  cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
                  S_keys_cpu_buf, S_rids_cpu_buf, S_key_indices_cpu_buf,
                  S_match_found_cpu_buf, bucket_key_rids_cpu_buf,
                  S_bucket_ids_cpu_buf, result_key_cpu_buf, result_rid_cpu_buf,
                  result_sid_cpu_buf, result_count_cpu_buf);
              cpu_queue.flush();

              // 두 device의 probe phase 완료 대기
              cl_event event_handles[2] = {probe_events[0](),
                                           probe_events[1]()};
              clWaitForEvents(2, event_handles);
              total_time += probe_timer.getTimeMilliseconds();
            }

            double avg_time = total_time / num_iterations;
            std::cout << "Ratio " << test_ratio << "%: Average = " << avg_time
                      << " ms (GPU: " << gpu_portion << ", CPU: " << cpu_portion
                      << ")" << std::endl;

            if (avg_time < best_avg_time) {
              best_avg_time = avg_time;
              best_ratio = test_ratio;
            }
          }

          std::cout << "\n=== Benchmark Results ===" << std::endl;
          std::cout << "Best WORK_RATIO_GPU: " << best_ratio << "%"
                    << std::endl;
          std::cout << "Best Average Time: " << best_avg_time << " ms"
                    << std::endl;
        } else {
          std::cout << "\n=== OpenCL Probe Phase ===" << std::endl;

          // Probe phase: S 데이터를 나눠서 처리 (work distribution)
          // WORK_RATIO_GPU에 따라 분할 (각 portion은 4096의 배수)
          // GPU portion 계산: WORK_RATIO_GPU 비율에 따라 계산하고 4096의 배수로
          // 조정
          size_t gpu_portion =
              ((S_LENGTH * WORK_RATIO_GPU / 100) / 4096) * 4096;
          // CPU portion 계산: 나머지를 4096의 배수로 조정
          size_t cpu_portion = ((S_LENGTH - gpu_portion) / 4096) * 4096;
          // 합이 정확히 S_LENGTH가 되도록 GPU portion 재조정
          gpu_portion = S_LENGTH - cpu_portion;

          // GPU용 sub-buffers 생성 (cl_buffer_region 사용)
          // GPU는 앞부분 처리 (0부터 gpu_portion까지)
          cl_buffer_region gpu_keys_region = {0,
                                              sizeof(uint32_t) * gpu_portion};
          cl_buffer_region gpu_rids_region = {0,
                                              sizeof(uint32_t) * gpu_portion};
          cl_buffer_region gpu_result_key_region = {
              0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region gpu_result_rid_region = {
              0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region gpu_result_sid_region = {
              0, sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region gpu_result_count_region = {0, sizeof(uint32_t) *
                                                             gpu_portion};

          cl::Buffer S_keys_gpu_buf = S_keys_buf.createSubBuffer(
              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &gpu_keys_region);
          cl::Buffer S_rids_gpu_buf = S_rids_buf.createSubBuffer(
              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &gpu_rids_region);
          cl::Buffer S_bucket_ids_gpu_buf(context, CL_MEM_READ_WRITE,
                                          sizeof(uint32_t) * gpu_portion);
          cl::Buffer S_key_indices_gpu_buf(context, CL_MEM_READ_WRITE,
                                           sizeof(int) * gpu_portion);
          cl::Buffer S_match_found_gpu_buf(context, CL_MEM_READ_WRITE,
                                           sizeof(uint32_t) * gpu_portion);
          cl::Buffer result_key_gpu_buf = result_key_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &gpu_result_key_region);
          cl::Buffer result_rid_gpu_buf = result_rid_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &gpu_result_rid_region);
          cl::Buffer result_sid_gpu_buf = result_sid_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &gpu_result_sid_region);
          cl::Buffer result_count_gpu_buf = result_count_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &gpu_result_count_region);

          // CPU용 sub-buffers 생성
          // CPU는 뒷부분 처리 (gpu_portion부터 S_LENGTH까지)
          cl_buffer_region cpu_keys_region = {sizeof(uint32_t) * gpu_portion,
                                              sizeof(uint32_t) * cpu_portion};
          cl_buffer_region cpu_rids_region = {sizeof(uint32_t) * gpu_portion,
                                              sizeof(uint32_t) * cpu_portion};
          cl_buffer_region cpu_result_key_region = {
              sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
              sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region cpu_result_rid_region = {
              sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
              sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region cpu_result_sid_region = {
              sizeof(uint32_t) * gpu_portion * MAX_RIDS_PER_KEY,
              sizeof(uint32_t) * cpu_portion * MAX_RIDS_PER_KEY};
          cl_buffer_region cpu_result_count_region = {
              sizeof(uint32_t) * gpu_portion, sizeof(uint32_t) * cpu_portion};

          cl::Buffer S_keys_cpu_buf = S_keys_buf.createSubBuffer(
              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &cpu_keys_region);
          cl::Buffer S_rids_cpu_buf = S_rids_buf.createSubBuffer(
              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &cpu_rids_region);
          cl::Buffer S_bucket_ids_cpu_buf(context, CL_MEM_READ_WRITE,
                                          sizeof(uint32_t) * cpu_portion);
          cl::Buffer S_key_indices_cpu_buf(context, CL_MEM_READ_WRITE,
                                           sizeof(int) * cpu_portion);
          cl::Buffer S_match_found_cpu_buf(context, CL_MEM_READ_WRITE,
                                           sizeof(uint32_t) * cpu_portion);
          cl::Buffer result_key_cpu_buf = result_key_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &cpu_result_key_region);
          cl::Buffer result_rid_cpu_buf = result_rid_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &cpu_result_rid_region);
          cl::Buffer result_sid_cpu_buf = result_sid_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &cpu_result_sid_region);
          cl::Buffer result_count_cpu_buf = result_count_buf.createSubBuffer(
              CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
              &cpu_result_count_region);

          opencl_timer.reset();
          cl::Event probe_events[2];

          // GPU probe phase - GPU hash table only
          p1(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
             S_keys_gpu_buf, S_bucket_ids_gpu_buf);

          // p2: check bucket validity
          p2(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
             S_bucket_ids_gpu_buf, bucket_total_cpu_buf);

          // p3: search key lists
          p3(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
             S_keys_gpu_buf, S_bucket_ids_gpu_buf, bucket_keys_cpu_buf,
             S_key_indices_gpu_buf, S_match_found_gpu_buf);

          // p4: join matching records
          probe_events[0] =
              p4(cl::EnqueueArgs(gpu_queue, cl::NDRange(gpu_portion)),
                 S_keys_gpu_buf, S_rids_gpu_buf, S_key_indices_gpu_buf,
                 S_match_found_gpu_buf, bucket_key_rids_cpu_buf,
                 S_bucket_ids_gpu_buf, result_key_gpu_buf, result_rid_gpu_buf,
                 result_sid_gpu_buf, result_count_gpu_buf);
          gpu_queue.flush();

          // CPU probe phase - CPU hash table only
          p1(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
             S_keys_cpu_buf, S_bucket_ids_cpu_buf);

          // p2: check bucket validity
          p2(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
             S_bucket_ids_cpu_buf, bucket_total_cpu_buf);

          // p3: search key lists
          p3(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
             S_keys_cpu_buf, S_bucket_ids_cpu_buf, bucket_keys_cpu_buf,
             S_key_indices_cpu_buf, S_match_found_cpu_buf);

          // p4: join matching records
          probe_events[1] =
              p4(cl::EnqueueArgs(cpu_queue, cl::NDRange(cpu_portion)),
                 S_keys_cpu_buf, S_rids_cpu_buf, S_key_indices_cpu_buf,
                 S_match_found_cpu_buf, bucket_key_rids_cpu_buf,
                 S_bucket_ids_cpu_buf, result_key_cpu_buf, result_rid_cpu_buf,
                 result_sid_cpu_buf, result_count_cpu_buf);
          cpu_queue.flush();

          // 두 device의 probe phase 완료 대기
          cl_event event_handles[2] = {probe_events[0](), probe_events[1]()};
          clWaitForEvents(2, event_handles);
          double probe_time = opencl_timer.getTimeMilliseconds();
          std::cout << "Probe Phase Total: " << probe_time << " ms"
                    << std::endl;
          std::cout << "OpenCL Hash Join Total: " << build_time + probe_time
                    << " ms" << std::endl;
          std::cout << "\nWork distribution: GPU " << gpu_portion
                    << " tuples, CPU " << cpu_portion << " tuples" << std::endl;

          // Read back result counts directly from shared buffer
          std::vector<uint32_t> result_counts(S_LENGTH);
          cpu_queue.enqueueReadBuffer(result_count_buf, CL_TRUE, 0,
                                      sizeof(uint32_t) * S_LENGTH,
                                      &result_counts[0]);

          // Calculate total number of results
          uint32_t num_results = 0;
          for (uint32_t i = 0; i < S_LENGTH; i++) {
            num_results += result_counts[i];
          }

          std::cout << "OpenCL produced " << num_results << " joined tuples"
                    << std::endl;

          std::vector<JoinedTuple> opencl_res;

          if (num_results > 0) {
            // Read the sparse result buffers
            std::vector<uint32_t> sparse_keys(max_result_size);
            std::vector<uint32_t> sparse_rids(max_result_size);
            std::vector<uint32_t> sparse_sids(max_result_size);

            cpu_queue.enqueueReadBuffer(result_key_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_keys[0]);
            cpu_queue.enqueueReadBuffer(result_rid_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_rids[0]);
            cpu_queue.enqueueReadBuffer(result_sid_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_sids[0]);

            // Compact results: remove empty slots
            std::vector<uint32_t> result_keys;
            std::vector<uint32_t> result_rids;
            std::vector<uint32_t> result_sids;
            result_keys.reserve(num_results);
            result_rids.reserve(num_results);
            result_sids.reserve(num_results);

            for (uint32_t i = 0; i < S_LENGTH; i++) {
              uint32_t count = result_counts[i];
              if (count > 0) {
                uint32_t base_offset = i * MAX_RIDS_PER_KEY;
                for (uint32_t j = 0; j < count; j++) {
                  result_keys.push_back(sparse_keys[base_offset + j]);
                  result_rids.push_back(sparse_rids[base_offset + j]);
                  result_sids.push_back(sparse_sids[base_offset + j]);
                }
              }
            }

            // Convert to JoinedTuple format
            opencl_res.reserve(num_results);
            for (uint32_t i = 0; i < num_results; i++) {
              JoinedTuple jt;
              jt.key = result_keys[i];
              jt.ridR = result_rids[i];
              jt.ridS = result_sids[i];
              opencl_res.push_back(jt);
            }
          }

          // Compare OpenCL result with Standard join result
          if (run_std_join && opencl_res.size() > 0) {
            bool opencl_pass = (opencl_res.size() == stdRes.size());
            if (opencl_pass) {
              std::unordered_map<uint32_t, uint64_t> openclKeyCount;
              std::unordered_map<uint32_t, uint64_t> stdKeyCount;
              openclKeyCount.reserve(R_LENGTH);
              stdKeyCount.reserve(R_LENGTH);
              for (const auto &jt : opencl_res)
                openclKeyCount[jt.key]++;
              for (const auto &jt : stdRes)
                stdKeyCount[jt.key]++;
              if (openclKeyCount.size() != stdKeyCount.size()) {
                opencl_pass = false;
              } else {
                for (const auto &kv : openclKeyCount) {
                  auto it = stdKeyCount.find(kv.first);
                  if (it == stdKeyCount.end() || it->second != kv.second) {
                    opencl_pass = false;
                    break;
                  }
                }
              }
            }
            std::cout << "OpenCL Verification: "
                      << (opencl_pass ? "PASS" : "FAIL") << "\n";
          }
        } // end of else block (normal mode probe phase)
      } else if (deviceIndex == 3) { // OL optimization
        cl::Device CPU = devices[0];
        cl::Device GPU = devices[1];

        std::string name;
        getDeviceName(CPU, name);
        std::cout << "\n=== OL Optimization Mode ===" << std::endl;
        std::cout << "Using OpenCL CPU: " << name << "\n";
        getDeviceName(GPU, name);
        std::cout << "Using OpenCL GPU: " << name << "\n";
        std::cout << "Step assignment: b1,b2,b3->CPU, b4->GPU, "
                  << "p1,p2,p3->CPU, p4->GPU\n"
                  << std::endl;

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(CPU);
        chosen_device.push_back(GPU);
        cl::Context context(chosen_device);
        cl::CommandQueue cpu_queue(context, CPU, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue gpu_queue(context, GPU, CL_QUEUE_PROFILING_ENABLE);

        cl::Program program(context, util::loadProgram("hj.cl"), true);
        cl::make_kernel<cl::Buffer, cl::Buffer> b1(program, "b1");
        cl::make_kernel<cl::Buffer, cl::Buffer> b2(program, "b2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(
            program, "b3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(
            program, "b4");
        cl::make_kernel<cl::Buffer, cl::Buffer> p1(program, "p1");
        cl::make_kernel<cl::Buffer, cl::Buffer> p2(program, "p2");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer>
            p3(program, "p3");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer>
            p4(program, "p4");

        std::vector<uint32_t> R_keys(R_LENGTH), R_rids(R_LENGTH),
            S_keys(S_LENGTH), S_rids(S_LENGTH);
        for (int i = 0; i < R_LENGTH; i++) {
          R_keys[i] = R[i].key;
          R_rids[i] = R[i].rid;
        }
        for (int i = 0; i < S_LENGTH; i++) {
          S_keys[i] = S[i].key;
          S_rids[i] = S[i].rid;
        }

        // buffer init
        cl::Buffer R_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_keys[0]);
        cl::Buffer S_keys_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_keys[0]);
        cl::Buffer R_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * R_LENGTH, &R_rids[0]);
        cl::Buffer S_rids_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              sizeof(uint32_t) * S_LENGTH, &S_rids[0]);

        // b1
        cl::Buffer R_bucket_ids_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * R_LENGTH);

        // b2
        cl::Buffer bucket_total_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * BUCKET_HEADER_NUMBER);

        // b3
        cl::Buffer bucket_keys_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET);
        cl::Buffer key_indices_buf(context, CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * R_LENGTH);

        // b4
        cl::Buffer bucket_key_rids_buf(
            context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET *
                MAX_RIDS_PER_KEY);

        // p1
        cl::Buffer S_bucket_ids_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        // p3
        cl::Buffer S_key_indices_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(int) * S_LENGTH);
        cl::Buffer S_match_found_buf(context,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t) * S_LENGTH);

        // p4
        size_t max_result_size = (size_t)S_LENGTH * MAX_RIDS_PER_KEY;
        cl::Buffer result_key_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_rid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_sid_buf(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(uint32_t) * max_result_size);
        cl::Buffer result_count_buf(context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(uint32_t) * S_LENGTH);

        std::vector<uint32_t> bucket_totalNumcounts(BUCKET_HEADER_NUMBER, 0);

        // Initialize buffers to zero
        cpu_queue.enqueueWriteBuffer(bucket_total_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER,
                                     &bucket_totalNumcounts[0]);

        // Zero-initialize large buffers
        std::vector<uint32_t> bucket_key_rids_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
            0xffffffffu);
        std::vector<uint32_t> bucket_keys_init(
            BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET, 0xffffffffu);
        std::vector<uint32_t> result_count_init(S_LENGTH, 0);
        cpu_queue.enqueueWriteBuffer(bucket_key_rids_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
                                     &bucket_key_rids_init[0]);
        cpu_queue.enqueueWriteBuffer(bucket_keys_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                                         MAX_KEYS_PER_BUCKET,
                                     &bucket_keys_init[0]);
        cpu_queue.enqueueWriteBuffer(result_count_buf, CL_TRUE, 0,
                                     sizeof(uint32_t) * S_LENGTH,
                                     &result_count_init[0]);

        if (run_bench) {
          std::cout << "\n=== OL Step Combination Benchmark ===" << std::endl;
          std::cout << "Testing all combinations of b3, b4, p3, p4\n";
          std::cout << "Format: [b3][b4][p3][p4] where 0=CPU, 1=GPU\n";
          std::cout << "Running 10 iterations per combination...\n"
                    << std::endl;

          double best_avg_time = 1e9;
          int best_combination = 0;

          // Test all 16 combinations (2^4)
          for (int combo = 0; combo < 16; combo++) {
            // Extract bit flags: b3, b4, p3, p4
            bool b3_on_gpu = (combo & 1) != 0; // bit 0
            bool b4_on_gpu = (combo & 2) != 0; // bit 1
            bool p3_on_gpu = (combo & 4) != 0; // bit 2
            bool p4_on_gpu = (combo & 8) != 0; // bit 3

            double total_time = 0.0;
            const int num_iterations = 10;

            for (int iter = 0; iter < num_iterations; iter++) {
              // Re-initialize buffers for each iteration
              cpu_queue.enqueueWriteBuffer(bucket_total_buf, CL_TRUE, 0,
                                           sizeof(uint32_t) *
                                               BUCKET_HEADER_NUMBER,
                                           &bucket_totalNumcounts[0]);
              cpu_queue.enqueueWriteBuffer(
                  bucket_key_rids_buf, CL_TRUE, 0,
                  sizeof(uint32_t) * BUCKET_HEADER_NUMBER *
                      MAX_KEYS_PER_BUCKET * MAX_RIDS_PER_KEY,
                  &bucket_key_rids_init[0]);
              cpu_queue.enqueueWriteBuffer(
                  bucket_keys_buf, CL_TRUE, 0,
                  sizeof(uint32_t) * BUCKET_HEADER_NUMBER * MAX_KEYS_PER_BUCKET,
                  &bucket_keys_init[0]);
              cpu_queue.enqueueWriteBuffer(result_count_buf, CL_TRUE, 0,
                                           sizeof(uint32_t) * S_LENGTH,
                                           &result_count_init[0]);

              util::Timer iteration_timer;
              iteration_timer.reset();

              // Build Phase
              // b1: CPU (always)
              b1(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_keys_buf,
                 R_bucket_ids_buf);

              // b2: CPU (always)
              b2(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)),
                 R_bucket_ids_buf, bucket_total_buf);

              // b3: conditional (CPU or GPU)
              if (b3_on_gpu) {
                b3(cl::EnqueueArgs(gpu_queue, cl::NDRange(R_LENGTH)),
                   R_keys_buf, R_bucket_ids_buf, bucket_keys_buf,
                   key_indices_buf);
              } else {
                b3(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)),
                   R_keys_buf, R_bucket_ids_buf, bucket_keys_buf,
                   key_indices_buf);
              }

              // b4: conditional (CPU or GPU)
              if (b4_on_gpu) {
                b4(cl::EnqueueArgs(gpu_queue, cl::NDRange(R_LENGTH)),
                   R_rids_buf, R_bucket_ids_buf, key_indices_buf,
                   bucket_key_rids_buf);
              } else {
                b4(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)),
                   R_rids_buf, R_bucket_ids_buf, key_indices_buf,
                   bucket_key_rids_buf);
              }

              // Probe Phase
              // p1: CPU (always)
              p1(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)), S_keys_buf,
                 S_bucket_ids_buf);

              // p2: CPU (always)
              p2(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)),
                 S_bucket_ids_buf, bucket_total_buf);

              // p3: conditional (CPU or GPU)
              if (p3_on_gpu) {
                p3(cl::EnqueueArgs(gpu_queue, cl::NDRange(S_LENGTH)),
                   S_keys_buf, S_bucket_ids_buf, bucket_keys_buf,
                   S_key_indices_buf, S_match_found_buf);
              } else {
                p3(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)),
                   S_keys_buf, S_bucket_ids_buf, bucket_keys_buf,
                   S_key_indices_buf, S_match_found_buf);
              }

              // p4: conditional (CPU or GPU)
              cl::Event p4_event;
              if (p4_on_gpu) {
                p4_event = p4(cl::EnqueueArgs(gpu_queue, cl::NDRange(S_LENGTH)),
                              S_keys_buf, S_rids_buf, S_key_indices_buf,
                              S_match_found_buf, bucket_key_rids_buf,
                              S_bucket_ids_buf, result_key_buf, result_rid_buf,
                              result_sid_buf, result_count_buf);
              } else {
                p4_event = p4(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)),
                              S_keys_buf, S_rids_buf, S_key_indices_buf,
                              S_match_found_buf, bucket_key_rids_buf,
                              S_bucket_ids_buf, result_key_buf, result_rid_buf,
                              result_sid_buf, result_count_buf);
              }
              cpu_queue.flush();
              gpu_queue.flush();
              p4_event.wait();
              total_time += iteration_timer.getTimeMilliseconds();
            }

            double avg_time = total_time / num_iterations;
            std::cout << "[" << (b3_on_gpu ? "1" : "0")
                      << (b4_on_gpu ? "1" : "0") << (p3_on_gpu ? "1" : "0")
                      << (p4_on_gpu ? "1" : "0") << "]: Average = " << avg_time
                      << " ms" << std::endl;

            if (avg_time < best_avg_time) {
              best_avg_time = avg_time;
              best_combination = combo;
            }
          }

          std::cout << "\n=== Benchmark Results ===" << std::endl;
          bool best_b3_gpu = (best_combination & 1) != 0;
          bool best_b4_gpu = (best_combination & 2) != 0;
          bool best_p3_gpu = (best_combination & 4) != 0;
          bool best_p4_gpu = (best_combination & 8) != 0;
          std::cout << "Best Combination: [" << (best_b3_gpu ? "1" : "0")
                    << (best_b4_gpu ? "1" : "0") << (best_p3_gpu ? "1" : "0")
                    << (best_p4_gpu ? "1" : "0") << "]" << std::endl;
          std::cout << "  b3: " << (best_b3_gpu ? "GPU" : "CPU") << std::endl;
          std::cout << "  b4: " << (best_b4_gpu ? "GPU" : "CPU") << std::endl;
          std::cout << "  p3: " << (best_p3_gpu ? "GPU" : "CPU") << std::endl;
          std::cout << "  p4: " << (best_p4_gpu ? "GPU" : "CPU") << std::endl;
          std::cout << "Best Average Time: " << best_avg_time << " ms"
                    << std::endl;
        } else {
          std::cout << "\n=== OpenCL Build Phase (OL) ===" << std::endl;
          util::Timer opencl_timer;
          opencl_timer.reset();

          // b1: compute hash bucket number - CPU
          std::cout << "b1: CPU" << std::endl;
          b1(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)), R_keys_buf,
             R_bucket_ids_buf);

          // b2: update bucket header - CPU
          std::cout << "b2: CPU" << std::endl;
          b2(cl::EnqueueArgs(cpu_queue, cl::NDRange(R_LENGTH)),
             R_bucket_ids_buf, bucket_total_buf);
          // b3: manage key lists - CPU
          std::cout << "b3: CPU" << std::endl;
          b3(cl::EnqueueArgs(gpu_queue, cl::NDRange(R_LENGTH)), R_keys_buf,
             R_bucket_ids_buf, bucket_keys_buf, key_indices_buf);

          // b4: insert record ids - GPU
          std::cout << "b4: GPU" << std::endl;
          cl::Event b4_event =
              b4(cl::EnqueueArgs(gpu_queue, cl::NDRange(R_LENGTH)), R_rids_buf,
                 R_bucket_ids_buf, key_indices_buf, bucket_key_rids_buf);

          // Probe Phase
          std::cout << "\n=== OpenCL Probe Phase (OL) ===" << std::endl;

          // p1: compute hash bucket number - CPU
          std::cout << "p1: CPU" << std::endl;
          p1(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)), S_keys_buf,
             S_bucket_ids_buf);

          // p2: check bucket validity - CPU
          std::cout << "p2: CPU" << std::endl;
          p2(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)),
             S_bucket_ids_buf, bucket_total_buf);
          // p3: search key lists - CPU
          std::cout << "p3: CPU" << std::endl;
          p3(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)), S_keys_buf,
             S_bucket_ids_buf, bucket_keys_buf, S_key_indices_buf,
             S_match_found_buf);

          // p4: join matching records - GPU
          std::cout << "p4: GPU" << std::endl;
          cl::Event p4_event =
              p4(cl::EnqueueArgs(cpu_queue, cl::NDRange(S_LENGTH)), S_keys_buf,
                 S_rids_buf, S_key_indices_buf, S_match_found_buf,
                 bucket_key_rids_buf, S_bucket_ids_buf, result_key_buf,
                 result_rid_buf, result_sid_buf, result_count_buf);

          cpu_queue.flush();
          gpu_queue.flush();
          p4_event.wait();

          double probe_time = opencl_timer.getTimeMilliseconds();
          std::cout << "OpenCL Hash Join Total: " << probe_time << " ms"
                    << std::endl;

          // Read back result counts
          std::vector<uint32_t> result_counts(S_LENGTH);
          cpu_queue.enqueueReadBuffer(result_count_buf, CL_TRUE, 0,
                                      sizeof(uint32_t) * S_LENGTH,
                                      &result_counts[0]);

          // Calculate total number of results
          uint32_t num_results = 0;
          for (uint32_t i = 0; i < S_LENGTH; i++) {
            num_results += result_counts[i];
          }

          std::cout << "OpenCL produced " << num_results << " joined tuples"
                    << std::endl;

          std::vector<JoinedTuple> opencl_res;

          if (num_results > 0) {
            // Read the sparse result buffers
            std::vector<uint32_t> sparse_keys(max_result_size);
            std::vector<uint32_t> sparse_rids(max_result_size);
            std::vector<uint32_t> sparse_sids(max_result_size);

            cpu_queue.enqueueReadBuffer(result_key_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_keys[0]);
            cpu_queue.enqueueReadBuffer(result_rid_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_rids[0]);
            cpu_queue.enqueueReadBuffer(result_sid_buf, CL_TRUE, 0,
                                        sizeof(uint32_t) * max_result_size,
                                        &sparse_sids[0]);

            // Compact results: remove empty slots
            std::vector<uint32_t> result_keys;
            std::vector<uint32_t> result_rids;
            std::vector<uint32_t> result_sids;
            result_keys.reserve(num_results);
            result_rids.reserve(num_results);
            result_sids.reserve(num_results);

            for (uint32_t i = 0; i < S_LENGTH; i++) {
              uint32_t count = result_counts[i];
              if (count > 0) {
                uint32_t base_offset = i * MAX_RIDS_PER_KEY;
                for (uint32_t j = 0; j < count; j++) {
                  result_keys.push_back(sparse_keys[base_offset + j]);
                  result_rids.push_back(sparse_rids[base_offset + j]);
                  result_sids.push_back(sparse_sids[base_offset + j]);
                }
              }
            }

            // Convert to JoinedTuple format
            opencl_res.reserve(num_results);
            for (uint32_t i = 0; i < num_results; i++) {
              JoinedTuple jt;
              jt.key = result_keys[i];
              jt.ridR = result_rids[i];
              jt.ridS = result_sids[i];
              opencl_res.push_back(jt);
            }
          }

          // Compare OpenCL result with Standard join result
          if (run_std_join && opencl_res.size() > 0) {
            bool opencl_pass = (opencl_res.size() == stdRes.size());
            if (opencl_pass) {
              std::unordered_map<uint32_t, uint64_t> openclKeyCount;
              std::unordered_map<uint32_t, uint64_t> stdKeyCount;
              openclKeyCount.reserve(R_LENGTH);
              stdKeyCount.reserve(R_LENGTH);
              for (const auto &jt : opencl_res)
                openclKeyCount[jt.key]++;
              for (const auto &jt : stdRes)
                stdKeyCount[jt.key]++;
              if (openclKeyCount.size() != stdKeyCount.size()) {
                opencl_pass = false;
              } else {
                for (const auto &kv : openclKeyCount) {
                  auto it = stdKeyCount.find(kv.first);
                  if (it == stdKeyCount.end() || it->second != kv.second) {
                    opencl_pass = false;
                    break;
                  }
                }
              }
            }
            std::cout << "OpenCL Verification: "
                      << (opencl_pass ? "PASS" : "FAIL") << "\n";
          }
        } // end of else block (normal mode)
      }
    } else { // run partitioned hash join
    }
  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
  }

  //===================== OpenCL Join End ==========================

  // Verification: Compare CPU join with Standard join
  if (run_cpu_join && run_std_join) {
    std::cout << "\n=== Verification (CPU vs Standard) ===" << std::endl;
    bool pass = (res.size() == stdRes.size());

    // Compare by per-key counts
    if (pass) {
      std::unordered_map<uint32_t, uint64_t> resKeyCount;
      std::unordered_map<uint32_t, uint64_t> stdKeyCount;
      resKeyCount.reserve(R_LENGTH);
      stdKeyCount.reserve(R_LENGTH);
      for (const auto &jt : res)
        resKeyCount[jt.key]++;
      for (const auto &jt : stdRes)
        stdKeyCount[jt.key]++;
      if (resKeyCount.size() != stdKeyCount.size()) {
        pass = false;
      } else {
        for (const auto &kv : resKeyCount) {
          auto it = stdKeyCount.find(kv.first);
          if (it == stdKeyCount.end() || it->second != kv.second) {
            pass = false;
            break;
          }
        }
      }
    }

    std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
  }

  return 0;
}
