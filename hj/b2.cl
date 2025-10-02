#define R_LENGTH 16777216

__kernel void b2(
    __global const uint* bucket_ids,
    __global uint* bucket_totalNum
) {
    uint gid = get_global_id(0);
    if(gid >= R_LENGTH) {
        return;
    }
}