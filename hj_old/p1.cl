// p1.cl - Probe Phase Step 1: Compute hash bucket number
// Each work-item processes one tuple from S table

#define RANGE 1024
#define GOLDEN_RATIO_32 2654435769U
#define BUCKET_HEADER_NUMBER 512
#define MAX_KEYS_PER_BUCKET 1024
#define MAX_RIDS_PER_KEY 16
#define MAX_VALUES_PER_TUPLE 16

// Hash function
uint hash_function(uint key) {
    return (key * GOLDEN_RATIO_32) % RANGE;
}

__kernel void p1_compute_hash(
    __global const uint* S_keys,      // Input: S table keys
    __global uint* hash_values,       // Output: computed hash values  
    __global uint* bucket_ids,        // Output: bucket IDs
    const uint length                  // Number of tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    // p1: compute hash bucket number
    uint h = hash_function(S_keys[gid]);
    uint id = h % BUCKET_HEADER_NUMBER;
    
    // Store results
    hash_values[gid] = h;
    bucket_ids[gid] = id;
}
