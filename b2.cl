// b2.cl - Build Phase Step 2: Visit the hash bucket header
// Each work-item updates the totalNum count for its assigned bucket

#define BUCKET_HEADER_NUMBER 64
#define MAX_KEYS_PER_BUCKET 1024

__kernel void b2_update_bucket_header(
    __global const uint* bucket_ids,     // Input: bucket IDs from b1
    __global uint* bucket_totalNum,      // Output: totalNum for each bucket
    const uint length                     // Number of tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    // b2: visit the hash bucket header
    uint bucket_id = bucket_ids[gid];
    
    // Atomically increment the total count for this bucket
    // Multiple work-items may access the same bucket simultaneously
    atomic_inc(&bucket_totalNum[bucket_id]);
}
