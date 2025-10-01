// b3.cl - Build Phase Step 3: Visit the hash key lists and create a key header if necessary
// Each work-item finds or creates a key entry in its bucket

#define BUCKET_HEADER_NUMBER 512
#define MAX_KEYS_PER_BUCKET 1024  // Maximum number of unique keys per bucket
#define MAX_RIDS_PER_KEY 16
#define MAX_VALUES_PER_TUPLE 16

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
    
    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;
    
    // Search entire bucket range to prevent race conditions
    // Use 0xFFFFFFFFU as empty slot marker (keys are 0~1023)
    for (int i = 0; i < MAX_KEYS_PER_BUCKET; i++) {
        uint current_key = bucket_keys[bucket_offset + i];
        
        if (current_key == key) {
            // Found existing key
            found = true;
            key_idx = i;
            break;
        } else if (current_key == 0xFFFFFFFFU) {
            // Found empty slot, try to claim it atomically
            if (atomic_cmpxchg(&bucket_keys[bucket_offset + i], 0xFFFFFFFFU, key) == 0xFFFFFFFFU) {
                // Successfully claimed this slot
                found = true;
                key_idx = i;
                // Update key count
                atomic_inc(&bucket_key_counts[bucket_id]);
                break;
            }
            // If atomic_cmpxchg failed, another work-item claimed this slot
            // Continue searching (that work-item might have inserted our key)
            i--; // Recheck this slot
        }
    }
    
    // If no slot was found (bucket full), key_idx remains -1
    
    // Store the key index for this tuple
    key_indices[gid] = key_idx;
}

