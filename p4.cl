// p4.cl - Probe Phase Step 4: Join matching records
// Each work-item joins its S tuple with all matching R tuples

#define BUCKET_HEADER_NUMBER 16
#define MAX_KEYS_PER_BUCKET 64
#define MAX_RIDS_PER_KEY 256
#define MAX_VALUES_PER_TUPLE 256    // Maximum values that can be stored in one tuple

__kernel void p4_join_records(
    __global const uint* S_values,       // Input: S table values
    __global const uint* bucket_ids,     // Input: bucket IDs from p1
    __global const int* key_indices,     // Input: key indices from p3
    __global const uint* match_found,    // Input: whether match was found from p3
    __global const uint* bucket_key_rids,  // Input: record IDs for each key from build phase
    __global const uint* bucket_key_rid_counts,  // Input: rid counts for each key
    __global uint* R_values,             // In/Out: R table values (to be updated)
    __global uint* R_value_counts,       // In/Out: number of values in each R tuple
    __global uint* join_results,         // Output: join result pairs [S_index, R_index]
    __global uint* join_count,           // Output: total number of joins performed
    const uint length                     // Number of S tuples to process
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= length) {
        return;
    }
    
    // Skip if no match was found
    if (match_found[gid] == 0) {
        return;
    }
    
    uint bucket_id = bucket_ids[gid];
    int key_idx = key_indices[gid];
    
    // Check if key index is valid
    if (key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
        return;
    }
    
    // p4: join matching records
    uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
    uint rid_count = bucket_key_rid_counts[bucket_key_offset];
    
    // Join with all R tuples that have the same key
    for (uint i = 0; i < rid_count; i++) {
        uint rid_offset = bucket_key_offset * MAX_RIDS_PER_KEY + i;
        uint r_tuple_id = bucket_key_rids[rid_offset];
        
        // Atomically get the next slot for this R tuple's values
        uint value_count = atomic_inc(&R_value_counts[r_tuple_id]);
        
        if (value_count < MAX_VALUES_PER_TUPLE) {
            // Add S value to R tuple
            uint r_value_offset = r_tuple_id * MAX_VALUES_PER_TUPLE + value_count;
            R_values[r_value_offset] = S_values[gid];
        }
        
        // Record this join in the results
        uint join_idx = atomic_inc(join_count);
        if (join_idx < length * MAX_RIDS_PER_KEY) {  // Prevent overflow
            join_results[join_idx * 2] = gid;          // S tuple index
            join_results[join_idx * 2 + 1] = r_tuple_id;  // R tuple index
        }
    }
}
