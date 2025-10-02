#define R_LENGTH 16777216
#define S_LENGTH 16777216

__kernel void b1(
    __global const uint* S_keys,
    __global uint* bucket_ids
) {
    uint gid = get_global_id(0);
    if(gid >= S_LENGTH) {
        return;
    }
    uint h = S_keys[gid] * 2654435769U % (R_LENGTH * 2);
    bucket_ids[gid] = h;
}