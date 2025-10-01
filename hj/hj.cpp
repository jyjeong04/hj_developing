#include "hj.hpp"
#include "param.hpp"
#include "datagen.cpp"
#include "util.hpp"
 

#include <cstddef>
#include <cstdint>
#include <iostream>
 
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cctype>
 
#include <ostream>
#include <vector>
 
#include <unordered_map>

static std::vector<JoinedTuple> run_standard_hash_join(const std::vector<Tuple>& R,
        const std::vector<Tuple>& S) {
    // Build hash table from R: key -> list of R rids
    std::unordered_map<uint32_t, std::vector<uint32_t>> rIndex;
    rIndex.reserve(static_cast<size_t>(R_LENGTH) * 2);
    for(const auto &t : R) {
        rIndex[t.key].push_back(t.rid);
    }

    // Probe with S and emit joins
    std::vector<JoinedTuple> out;
    // Rough reservation heuristic to reduce reallocations
    out.reserve(static_cast<size_t>(S_LENGTH) / 4);
    for(const auto &s : S) {
        auto it = rIndex.find(s.key);
        if(it == rIndex.end()) continue;
        const std::vector<uint32_t>& rids = it->second;
        for(size_t i = 0; i < rids.size(); i++) {
            JoinedTuple jt;
            jt.key = s.key;
            jt.ridR = rids[i];
            jt.ridS = s.rid;
            out.push_back(jt);
        }
    }
    return out;
}

uint32_t hash(uint32_t key) {
    return (key * 2654435769U) % (R_LENGTH * 2);
}

int main() {
    srand(time(NULL));
    std::vector<BucketHeader> bucketList(R_LENGTH * 2);
    // std::vector<Tuple> R = RGenerator();
    // std::vector<Tuple> S = SGenerator(R);
    std::vector<Tuple> R(R_LENGTH);
    std::vector<Tuple> S(S_LENGTH);
    int i;
    for(i = 0; i < R_LENGTH; i++) {
        R[i].key = rand() % R_LENGTH;
        R[i].rid = rand() % 1024;
    }
    for(i = 0; i < S_LENGTH; i++) {
        S[i].key = rand() % R_LENGTH;
        S[i].rid = rand() % 1024;
    }
    std::vector<JoinedTuple> res;
    res.reserve(S_LENGTH / 4);

    util::Timer timer;

    timer.reset();
    for(i = 0; i < R_LENGTH; i++) {
        // b1: compute hash bucket number
        uint32_t id = hash(R[i].key);
        // b2: visit the hash bucket header
        BucketHeader &tmpHeader = bucketList[id];
        tmpHeader.totalNum++;
        // b3: visit the hash key lists and create a key header if necessary
        int j = 0;
        bool found = false;
        for(j = 0; j < tmpHeader.totalNum - 1; j++) {
            if(R[i].key == tmpHeader.keyList[j].key) {
                found = true;
                break;
            }
        }
        
        if(!found) {
            KeyHeader newKey;
            newKey.key = R[i].key;
            tmpHeader.keyList.push_back(newKey);
            j = tmpHeader.keyList.size() - 1;
        }
        
        // b4: insert the rid into the rid list
        tmpHeader.keyList[j].ridList.push_back(R[i].rid);
    }
    double step = timer.getTimeMilliseconds();
    timer.reset();
    for(i = 0; i < S_LENGTH; i++) {
        // p1: compute hash bucket number
        uint32_t id = hash(S[i].key);
        // p2: visit the hash bucket header
        BucketHeader &tmpHeader = bucketList[id];
        // p3: visit the hash key lists
        int j = 0;
        bool found = false;
        for(j = 0; j < tmpHeader.totalNum; j++) {
            if(S[i].key == tmpHeader.keyList[j].key) {
                found = true;
                break;
            }
        }

        // p4: visit the matching build tuple to compare keys and produce output tuple
        if(found) {
            for(int h = 0; h < tmpHeader.keyList[j].ridList.size(); h++) {
                JoinedTuple t;
                t.key = S[i].key;
                t.ridR = tmpHeader.keyList[j].ridList[h];
                t.ridS = S[i].rid;
                res.push_back(t);
            }
        }
    }

    std::cout << timer.getTimeMilliseconds() << "ms" << std::endl;
    std::cout << step << "ms\n";

    // Run standard hash join and verify against hash-join result
    util::Timer timer2;
    timer2.reset();
    std::vector<JoinedTuple> stdRes = run_standard_hash_join(R, S);
    double stdMs = timer2.getTimeMilliseconds();
    std::cout << "std_join: " << stdRes.size() << " tuples, " << stdMs << "ms\n";

    // Compare by total count
    bool pass = (res.size() == stdRes.size());

    // Compare by per-key counts
    if(pass) {
        std::unordered_map<uint32_t, uint64_t> resKeyCount;
        std::unordered_map<uint32_t, uint64_t> stdKeyCount;
        resKeyCount.reserve(R_LENGTH);
        stdKeyCount.reserve(R_LENGTH);
        for(const auto &jt : res) resKeyCount[jt.key]++;
        for(const auto &jt : stdRes) stdKeyCount[jt.key]++;
        if(resKeyCount.size() != stdKeyCount.size()) {
            pass = false;
        } else {
            for(const auto &kv : resKeyCount) {
                auto it = stdKeyCount.find(kv.first);
                if(it == stdKeyCount.end() || it->second != kv.second) { pass = false; break; }
            }
        }
    }

    std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";

//     // Print a few sample results (up to 20)
//     std::cout << "RESULT (first up to 20 tuples)\n";
//     int printN = static_cast<int>(std::min<size_t>(20, res.size()));
//     for(int k = 0; k < printN; k++) {
//         std::cout << "(key=" << res[k].key
//                   << ", ridR=" << res[k].ridR
//                   << ", ridS=" << res[k].ridS << ")\n";
//     }
}

