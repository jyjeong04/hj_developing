// p2.cl - Probe Phase Step 2: Visit the hash bucket header
// Each work-item updates the totalNum count for its assigned bucket (optional for probe phase)

#define BUCKET_HEADER_NUMBER 512
#define MAX_KEYS_PER_BUCKET 1024
#define MAX_RIDS_PER_KEY 16
#define MAX_VALUES_PER_TUPLE 16

__kernel void p2_update_bucket_header(
    __global const uint* bucket_ids,     // Input: bucket IDs from p1
    __global uint* bucket_totalNum,      // Output: totalNum for each bucket
    const uint length                     // Number of tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    // p2: visit the hash bucket header
    uint bucket_id = bucket_ids[gid];
    
    // Atomically increment the total count for this bucket
    // Note: This step is optional in probe phase, mainly for statistics
    atomic_inc(&bucket_totalNum[bucket_id]);
}
