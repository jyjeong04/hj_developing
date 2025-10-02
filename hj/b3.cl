#define R_LENGTH 16777216
#define MAX_KEYS_PER_BUCKET 4

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
        if(current_key == key) {
            key_idx = i;
            break;
        } else if (current_key == 0) {
            if(atomic_cmpxchg(&bucket_keys[bucket_offset + i], 0, key) == 0) {
                atomic_inc(&bucket_key_counts[bucket_id]);
                key_idx = i;
                break;
            }
            i--;
        }
    }

    key_indices[gid] = key_idx;
}