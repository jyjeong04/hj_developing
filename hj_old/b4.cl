// b4.cl - Build Phase Step 4: Insert the record id into the rid list
// Each work-item adds its record ID to the appropriate key's rid list

#define BUCKET_HEADER_NUMBER 512
#define MAX_KEYS_PER_BUCKET 1024
#define MAX_RIDS_PER_KEY 16    // Maximum number of record IDs per key
#define MAX_VALUES_PER_TUPLE 16

__kernel void b4_insert_record_ids(
    __global const uint* bucket_ids,      // Input: bucket IDs from b1
    __global const int* key_indices,      // Input: key indices from b3
    __global uint* bucket_key_rids,       // Output: record IDs for each key [BUCKET][KEY][RID_INDEX]
    __global uint* bucket_key_rid_counts, // Output: number of record IDs for each key [BUCKET][KEY]
    const uint length                      // Number of tuples to process  
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    uint bucket_id = bucket_ids[gid];
    int key_idx = key_indices[gid];
    
    // Check if key index is valid
    if (key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
        return; // Invalid key index, skip
    }
    
    // b4: insert the record id into the rid list
    uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
    
    // Atomically get the next available slot for this key's rid list
    uint rid_count = atomic_inc(&bucket_key_rid_counts[bucket_key_offset]);
    
    if (rid_count < MAX_RIDS_PER_KEY) {
        // Calculate offset in the 3D array
        uint rid_offset = bucket_key_offset * MAX_RIDS_PER_KEY + rid_count;
        bucket_key_rids[rid_offset] = gid;  // Store the record ID (tuple index)
    }
    // If rid list is full, the increment is lost (error condition)
}
