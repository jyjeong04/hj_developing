#define R_LENGTH 16777216
#define S_LENGTH 16777216
#define MAX_KEYS_PER_BUCKET 4
#define MAX_RIDS_PER_KEY 4

__kernel void p4(
    __global const uint* S_keys,
    __global const uint* S_rids,
    __global const int* key_indices,
    __global const uint* match_found,
    __global const uint* bucket_key_rids,
    __global const uint* bucket_key_rid_counts,
    __global uint* result_key,
    __global uint* result_rid,
    __global uint* result_sid,
    __global uint* result_idx
) {
    uint gid = get_global_id(0);
    if(gid >= S_LENGTH) {
        return;
    }
    if(match_found[gid] == 0) {
        return;
    }
    int key_idx = key_indices[gid];
    if(key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
        return;
    }
    uint bucket_id = bucket_ids[gid];

    uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
    if(bucket_key_offset >= R_LENGTH * 2 * MAX_KEYS_PER_BUCKET) {
        return;
    }

    uint rid_count = bucket_key_rid_counts[bucket_key_offset];

    if(rid_count > MAX_RIDS_PER_KEY) {
        rid_count = MAX_RIDS_PER_KEY;
    }

    for (uint i = 0; i < rid_count; i++) {
        id = atomic_inc(&result_idx);
        result_key[id] = S_keys[gid];
        result_rid[id] = bucket_key_rids[bucket_key_offset * MAX_RIDS_PER_KEY + i];
        result_sid[id] = S_rids[gid];
    }
}