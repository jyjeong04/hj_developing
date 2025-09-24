// b3.cl - Build Phase Step 3: Visit the hash key lists and create a key header if necessary
// Each work-item finds or creates a key entry in its bucket

#define BUCKET_HEADER_NUMBER 16
#define MAX_KEYS_PER_BUCKET 64  // Maximum number of unique keys per bucket

__kernel void b3_manage_key_lists(
    __global const uint* R_keys,         // Input: R table keys
    __global const uint* bucket_ids,     // Input: bucket IDs from b1  
    __global uint* bucket_keys,          // Output: keys in each bucket [BUCKET_HEADER_NUMBER][MAX_KEYS_PER_BUCKET]
    __global uint* bucket_key_counts,    // Output: number of unique keys in each bucket
    __global int* key_indices,           // Output: index of key for each tuple in its bucket
    const uint length                     // Number of tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    uint key = R_keys[gid];
    uint bucket_id = bucket_ids[gid];
    
    // b3: visit the hash key lists and create a key header if necessary
    bool found = false;
    int key_idx = -1;
    
    // Search for existing key in this bucket
    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;
    uint current_key_count = bucket_key_counts[bucket_id]; // Simple read (OpenCL 3.0 compatible)
    
    for (int i = 0; i < current_key_count; i++) {
        if (bucket_keys[bucket_offset + i] == key) {
            found = true;
            key_idx = i;
            break;
        }
    }
    
    // If key not found, try to add it
    if (!found) {
        // Atomically get the next available slot
        uint old_count = atomic_inc(&bucket_key_counts[bucket_id]);
        
        if (old_count < MAX_KEYS_PER_BUCKET) {
            // We got a valid slot
            bucket_keys[bucket_offset + old_count] = key;
            key_idx = old_count;
        }
        // If bucket is full, key_idx remains -1 (error condition)
    }
    
    // Store the key index for this tuple
    key_indices[gid] = key_idx;
}
