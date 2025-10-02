// b1: compute hash bucket number

#define R_LENGTH 16777216
#define S_LENGTH 16777216

__kernel void b1(
    __global const uint* R_keys,
    __global uint* bucket_ids
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }
    uint h = R_keys[gid] * 2654435769U % (R_LENGTH * 2);
    bucket_ids[gid] = h;
}