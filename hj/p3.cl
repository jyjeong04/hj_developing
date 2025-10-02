#define S_LENGTH 16777216
#define MAX_KEYS_PER_BUCKET 4

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

    uint key = S_keys[gid];
    uint bucket_id = bucket_ids[gid];

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