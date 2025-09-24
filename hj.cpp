#include <vector>
#include <cstdint>

#define RANGE 1000
#define KEY_RANGE 100
static const uint32_t GOLDEN_RATIO_32 = 2654435769U;

uint32_t hash(uint32_t key) {
    return (key * GOLDEN_RATIO_32) % RANGE;
}

struct tuple {
    uint32_t key;
    uint32_t value;
};

struct keyList {
    uint32_t key;
    std::vector<uint32_t> rid;
    keyList* next;
};

struct bucketHeader {
    int totalNum;
    std::vector<keyList> kl;
};

int main(void) {
    // std::vector<tuple> 
    return 0;
}
