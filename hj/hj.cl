#include "param.hpp"

// b1: compute hash bucket number
__kernel void b1(
    __global const uint* R_keys,
    __global uint* bucket_ids
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }
    uint h = R_keys[gid] * HASH_SEED % (BUCKET_HEADER_NUMBER);
    bucket_ids[gid] = h;
}

__kernel void b2(
    __global const uint* bucket_ids,
    __global uint* bucket_total
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }
    uint bucket_id = bucket_ids[gid];
    atomic_inc(&bucket_total[bucket_id]);
}

__kernel void b3(
    __global const uint* R_keys,
    __global const uint* bucket_ids,
    __global uint* bucket_keys,
    __global uint* bucket_key_counts,
    __global int* key_indices
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }

    uint key = R_keys[gid];
    uint bucket_id = bucket_ids[gid];
    int key_idx = -1;

    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;
    for (int i = 0; i < MAX_KEYS_PER_BUCKET; i++) {
        uint current_key = bucket_keys[bucket_offset + i];

        // if(current_key == 0) {
        //     if(atomic_cmpxchg(&bucket_keys[bucket_offset + i], 0, key) == 0) {
        //         atomic_inc(&bucket_key_counts[bucket_id]);
        //         key_idx = i;
        //         break;
        //     }
        //     i--;
        // } else if (current_key == key) {
        //     key_idx = i;
        //     break;
        // }

        if(current_key == key) {
            key_idx = i;
            break;
        } else if (current_key == 0xffffffffu) {
            if(atomic_cmpxchg(&bucket_keys[bucket_offset + i], 0xffffffffu, key) == 0xffffffffu) {
                atomic_inc(&bucket_key_counts[bucket_id]);
                key_idx = i;
                break;
            }
            i--;
        }
    }

    key_indices[gid] = key_idx;
}

__kernel void b4(
    __global const uint* R_rids,
    __global const uint* bucket_ids,
    __global const int* key_indices,
    __global uint* bucket_key_rids,
    __global uint* bucket_key_rid_counts
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }

    uint bucket_id = bucket_ids[gid];
    int key_idx = key_indices[gid];

    if(key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
        return;
    }

    uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
    uint rid_count = atomic_inc(&bucket_key_rid_counts[bucket_key_offset]);

    if(rid_count < MAX_RIDS_PER_KEY) {
        bucket_key_rids[bucket_key_offset * MAX_RIDS_PER_KEY + rid_count] = R_rids[gid];
    }
}

__kernel void p1(
    __global const uint* S_keys,
    __global uint* bucket_ids
) {
    uint gid = get_global_id(0);
    if(gid >= S_LENGTH) {
        return;
    }
    uint h = S_keys[gid] * HASH_SEED % (BUCKET_HEADER_NUMBER);
    bucket_ids[gid] = h;
}

__kernel void p2(
    __global uint* bucket_ids,
    __global uint* bucket_total
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= S_LENGTH) {
        return;
    }
    uint bucket_id = bucket_ids[gid];
    if(!bucket_total[bucket_id]) {
        bucket_ids[gid] = 0xffffffff;
    }
}

__kernel void p3(
    __global const uint* S_keys,
    __global const uint* bucket_ids,
    __global const uint* bucket_keys,
    __global const uint* bucket_key_counts,
    __global int* key_indices,
    __global uint* match_found
) {
    uint gid = get_global_id(0);
    if(gid >= S_LENGTH) {
        return;
    }
    uint bucket_id = bucket_ids[gid];
    if(bucket_id == 0xffffffff) {
        return;
    }

    uint key = S_keys[gid];

    bool found = false;
    int key_idx = -1;

    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;

    for (int i = 0; i < bucket_key_counts[bucket_id]; i++) {
        uint bucket_key = bucket_keys[bucket_offset + i];
        if(bucket_key == key) {
            found = true;
            key_idx = i;
            break;
        }
    }
    key_indices[gid] = key_idx;
    match_found[gid] = found ? 1 : 0;
}

__kernel void p4(
    __global const uint* S_keys,
    __global const uint* S_rids,
    __global const int* key_indices,
    __global const uint* match_found,
    __global const uint* bucket_key_rids,
    __global const uint* bucket_key_rid_counts,
    __global const uint* bucket_ids,
    __global uint* result_key,
    __global uint* result_rid,
    __global uint* result_sid,
    __global uint* result_idx
) {
    uint gid = get_global_id(0);
    if(gid >= S_LENGTH) {
        return;
    }
    uint bucket_id = bucket_ids[gid];
    if(bucket_id == 0xffffffff) {
        return;
    }
    if(match_found[gid] == 0) {
        return;
    }
    int key_idx = key_indices[gid];
    if(key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
        return;
    }

    uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
    if(bucket_key_offset >= R_LENGTH * 2 * MAX_KEYS_PER_BUCKET) {
        return;
    }

    uint rid_count = bucket_key_rid_counts[bucket_key_offset];

    if(rid_count > MAX_RIDS_PER_KEY) {
        rid_count = MAX_RIDS_PER_KEY;
    }

    for (uint i = 0; i < rid_count; i++) {
        uint id = atomic_inc(result_idx);
        result_key[id] = S_keys[gid];
        result_rid[id] = bucket_key_rids[bucket_key_offset * MAX_RIDS_PER_KEY + i];
        result_sid[id] = S_rids[gid];
    }
}