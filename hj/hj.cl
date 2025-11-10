#include "param.hpp"

// b1: compute hash bucket number
__kernel void b1(__global const uint *R_keys, __global uint *bucket_ids) {
  uint gid = get_global_id(0);
  if (gid >= R_LENGTH) {
    return;
  }
  uint h = R_keys[gid] * HASH_SEED % (BUCKET_HEADER_NUMBER);
  bucket_ids[gid] = h;
}

__kernel void b2(__global const uint *bucket_ids, __global uint *bucket_total) {
  uint gid = get_global_id(0);
  if (gid >= R_LENGTH) {
    return;
  }
}

__kernel void b3(__global const uint *R_keys, __global uint *bucket_ids,
                 __global uint *bucket_keys, __global int *key_indices) {
  uint gid = get_global_id(0);
  if (gid >= R_LENGTH) {
    return;
  }

  uint key = R_keys[gid];
  uint original_bucket_id = bucket_ids[gid];
  uint bucket_id = original_bucket_id;
  int key_idx = -1;

  // Linear probing: search current bucket, if full move to next bu
  for (uint probe = 0; probe < BUCKET_HEADER_NUMBER; probe++) {
    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;

    // Search current bucket for existing key or empty slot
    for (int i = 0; i < MAX_KEYS_PER_BUCKET; i++) {
      uint current_key = bucket_keys[bucket_offset + i];

      if (current_key == key) {
        // Found existing key
        key_idx = i;
        break;
      } else if (current_key == 0xffffffffu) {
        // Found empty slot, try to claim it with retry mechanism
        // Instead of atomic_cmpxchg, use retry loop with verification
        bool claimed = false;
        const int max_retries = 10; // Maximum retry attempts

        for (int retry = 0; retry < max_retries; retry++) {
          // Double-check: read again to ensure slot is still empty
          uint check_key = bucket_keys[bucket_offset + i];

          if (check_key == key) {
            // Another thread inserted our key
            key_idx = i;
            claimed = true;
            break;
          } else if (check_key == 0xffffffffu) {
            // Slot is still empty, try to write our key
            bucket_keys[bucket_offset + i] = key;

            // Verify: read back to check if we successfully claimed it
            uint verify_key = bucket_keys[bucket_offset + i];

            if (verify_key == key) {
              // Successfully claimed this slot
              key_idx = i;
              claimed = true;
              break;
            }
            // else: Another thread wrote a different key, retry
          } else {
            // Slot is no longer empty (different key), break retry loop
            break;
          }
        }

        if (claimed) {
          break;
        }
        // Failed to claim after retries, continue searching in current bucket
      }
    }

    if (key_idx != -1) {
      // Found or inserted key successfully
      // Update bucket_id if it changed due to linear probing
      bucket_ids[gid] = bucket_id;
      break;
    }

    // Current bucket is full, move to next bucket (linear probing)
    bucket_id = (bucket_id + 1) % BUCKET_HEADER_NUMBER;
  }

  key_indices[gid] = key_idx;
}

__kernel void b4(__global const uint *R_rids, __global const uint *bucket_ids,
                 __global const int *key_indices,
                 __global uint *bucket_key_rids) {
  uint gid = get_global_id(0);
  if (gid >= R_LENGTH) {
    return;
  }
  uint rid = R_rids[gid];
  uint bucket_id = bucket_ids[gid];
  int key_idx = key_indices[gid];
  if (key_idx < 0 || key_idx >= MAX_KEYS_PER_BUCKET) {
    return;
  }

  uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;
  for (int i = 0; i < MAX_RIDS_PER_KEY; i++) {
    int tmp = bucket_key_offset * MAX_RIDS_PER_KEY + i;
    if (bucket_key_rids[tmp] == 0xffffffffu) {
      // Found empty slot, try to claim it with retry mechanism
      // Instead of atomic_cmpxchg, use retry loop with verification
      bool claimed = false;
      const int max_retries = 10; // Maximum retry attempts

      for (int retry = 0; retry < max_retries; retry++) {
        // Double-check: read again to ensure slot is still empty
        uint check_rid = bucket_key_rids[tmp];

        if (check_rid == rid) {
          // Another thread inserted our rid
          claimed = true;
          break;
        } else if (check_rid == 0xffffffffu) {
          // Slot is still empty, try to write our rid
          bucket_key_rids[tmp] = rid;

          // Verify: read back to check if we successfully claimed it
          uint verify_rid = bucket_key_rids[tmp];

          if (verify_rid == rid) {
            // Successfully claimed this slot
            claimed = true;
            break;
          }
          // else: Another thread wrote a different rid, retry
        } else {
          // Slot is no longer empty (different rid), break retry loop
          break;
        }
      }

      if (claimed) {
        return;
      }
      // Failed to claim after retries, continue to next slot
    }
  }
}

__kernel void p1(__global const uint *S_keys, __global uint *bucket_ids) {
  uint gid = get_global_id(0);
  if (gid >= S_LENGTH) {
    return;
  }
  uint h = S_keys[gid] * HASH_SEED % (BUCKET_HEADER_NUMBER);
  bucket_ids[gid] = h;
}

__kernel void p2(__global uint *bucket_ids, __global uint *bucket_total) {
  uint gid = get_global_id(0);

  // Check bounds
  if (gid >= S_LENGTH) {
    return;
  }
}

__kernel void p3(__global const uint *S_keys, __global uint *bucket_ids,
                 __global const uint *bucket_keys, __global int *key_indices,
                 __global uint *match_found) {
  uint gid = get_global_id(0);
  if (gid >= S_LENGTH) {
    return;
  }
  uint original_bucket_id = bucket_ids[gid];
  uint key = S_keys[gid];
  uint bucket_id = original_bucket_id;
  bool found = false;
  int key_idx = -1;

  // Linear probing: search current bucket, if not found move to next bucket
  for (uint probe = 0; probe < BUCKET_HEADER_NUMBER; probe++) {
    uint bucket_offset = bucket_id * MAX_KEYS_PER_BUCKET;

    // Search until we find the key or hit an empty slot (0xffffffffu)
    for (int i = 0; i < MAX_KEYS_PER_BUCKET; i++) {
      uint bucket_key = bucket_keys[bucket_offset + i];
      if (bucket_key == 0xffffffffu) {
        // Empty slot means key doesn't exist in this bucket
        break;
      }
      if (bucket_key == key) {
        found = true;
        key_idx = i;
        // Update bucket_id if it changed due to linear probing
        bucket_ids[gid] = bucket_id;
        break;
      }
    }

    if (found) {
      break;
    }

    // Key not found in current bucket, move to next bucket
    bucket_id = (bucket_id + 1) % BUCKET_HEADER_NUMBER;

    // Stop if we've wrapped around to the original bucket
    if (bucket_id == original_bucket_id && probe > 0) {
      break;
    }
  }

  key_indices[gid] = key_idx;
  match_found[gid] = found ? 1 : 0;
}

__kernel void p4(__global const uint *S_keys, __global const uint *S_rids,
                 __global const int *key_indices,
                 __global const uint *match_found,
                 __global const uint *bucket_key_rids,
                 __global const uint *bucket_ids, __global uint *result_key,
                 __global uint *result_rid, __global uint *result_sid,
                 __global uint *result_count) {
  uint gid = get_global_id(0);
  if (gid >= S_LENGTH) {
    return;
  }

  // Early exit if no match (result_count already initialized to 0)
  if (match_found[gid] == 0) {
    return;
  }

  uint bucket_id = bucket_ids[gid];
  int key_idx = key_indices[gid];

  if (key_idx < 0) {
    return;
  }

  uint bucket_key_offset = bucket_id * MAX_KEYS_PER_BUCKET + key_idx;

  // Each thread writes to its pre-allocated space: NO ATOMIC OPERATIONS
  // gid * MAX_RIDS_PER_KEY is the base offset for this thread
  uint base_offset = gid * MAX_RIDS_PER_KEY;
  uint s_key = S_keys[gid];
  uint s_rid = S_rids[gid];
  uint i;
  for (i = 0; i < MAX_RIDS_PER_KEY; i++) {
    uint rid = bucket_key_rids[bucket_key_offset * MAX_RIDS_PER_KEY + i];
    if (rid == 0xffffffffu)
      break;
    result_rid[base_offset + i] = rid;
    result_key[base_offset + i] = s_key;
    result_sid[base_offset + i] = s_rid;
  }
  result_count[gid] = i;
}
