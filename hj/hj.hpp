#pragma once

#include "param.hpp"
#include "cl.hpp"
#include <CL/cl_platform.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <vector>
#include <memory>

struct Tuple {
    uint32_t key;
    uint32_t rid;

    friend std::ostream &operator<<(std::ostream &out, const Tuple &tuple);
};

struct JoinedTuple {
    uint32_t key;
    uint32_t ridR;
    uint32_t ridS;

    friend std::ostream &operator<<(std::ostream &out, const JoinedTuple &tuple);
};

struct KeyHeader {
    uint32_t key{0};
    std::vector<uint32_t> ridList;
};

struct BucketHeader {
    uint32_t totalNum{0};
    std::vector<KeyHeader> keyList;
};

inline std::ostream &operator<<(std::ostream &out, const Tuple &tuple) {
    out << "Tuple{" << tuple.key << "," << tuple.rid << "}";
    return out;
}

inline std::ostream &operator<<(std::ostream &out, const JoinedTuple &tuple) {
    out << "Joined{" << tuple.key << ", R:" << tuple.ridR << ", S:" << tuple.ridS << "}";
    return out;
}