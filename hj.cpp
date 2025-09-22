#include <iostream>
#include <cstdint>

static const uint32_t GOLDEN_RATIO_32 = 2654435769U;

uint32_t hash(uint32_t key, uint32_t range) {
    return (key * GOLDEN_RATIO_32) % range;
}