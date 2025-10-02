#define R_LENGTH 16777216
#define MAX_KEYS_PER_BUCKET 4
#define MAX_RIDS_PER_KEY 4

__kernel void b4(
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
        uint rid_offset = bucket_key_offset * MAX_RIDS_PER_KEY + rid_count;
        bucket_key_rids[rid_offset] = gid;
    }
}