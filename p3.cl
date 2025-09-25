// p3.cl - Probe Phase Step 3: Visit the hash key lists and search for matching keys
// Each work-item searches for its key in the corresponding bucket

#define BUCKET_HEADER_NUMBER 512
#define MAX_KEYS_PER_BUCKET 1024

__kernel void p3_search_key_lists(
    __global const uint* S_keys,         // Input: S table keys
    __global const uint* bucket_ids,     // Input: bucket IDs from p1  
    __global const uint* bucket_keys,    // Input: keys in each bucket from build phase
    __global const uint* bucket_key_counts,  // Input: number of unique keys in each bucket
    __global int* key_indices,           // Output: index of matching key for each S tuple (-1 if not found)
    __global uint* match_found,          // Output: 1 if match found, 0 otherwise
    const uint length                     // Number of tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    uint key = S_keys[gid];
    uint bucket_id = bucket_ids[gid];
    
    // p3: visit the hash key lists and search for matching key
    bool found = false;
    int key_idx = -1;
    
    // Search for matching key in this bucket
    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;
    
    // Search entire bucket range until we find the key or hit an empty slot
    for (int i = 0; i < MAX_KEYS_PER_BUCKET; i++) {
        uint bucket_key = bucket_keys[bucket_offset + i];
        
        if (bucket_key == 0xFFFFFFFFU) {
            // Empty slot - no more keys to search
            break;
        } else if (bucket_key == key) {
            found = true;
            key_idx = i;
            break;
        }
    }
    
    // Store results
    key_indices[gid] = key_idx;
    match_found[gid] = found ? 1 : 0;
}
