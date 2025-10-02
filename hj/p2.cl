#define S_LENGTH 16777216

__kernel void p2(
    __global const uint* bucket_ids,
    __global uint* bucket_totalNum,
) {
    uint gid = get_global_id(0);
    
    // Check bounds
    if (gid >= S_LENGTH) {
        return;
    }
}
