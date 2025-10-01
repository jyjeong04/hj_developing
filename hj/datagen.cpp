#include "hj.hpp"
#include "param.hpp"
#include <random>
#include <unordered_set>
#include <algorithm>
#include <chrono>

std::vector<Tuple> RGenerator(){
	// RNG 준비: 고품질 시드 + 32비트 균등분포
	static std::mt19937 rng(
		static_cast<uint32_t>(
			std::chrono::high_resolution_clock::now().time_since_epoch().count()
		)
	);
	std::uniform_int_distribution<uint32_t> dist32(0u, 0xFFFFFFFFu);

	std::vector<Tuple> R(R_LENGTH);
	for(int i = 0; i < R_LENGTH; i++) {
		R[i].key = dist32(rng);
		R[i].rid = dist32(rng);
	}

	return R;
}

std::vector<Tuple> SGenerator(const std::vector<Tuple>& R){
	// R의 유니크 키를 수집
	std::unordered_set<uint32_t> uniq;
	uniq.reserve(static_cast<size_t>(R.size() * 1.3));
	for(const auto& t : R) uniq.insert(t.key);

	std::vector<uint32_t> keys;
	keys.reserve(uniq.size());
	for(uint32_t k : uniq) keys.push_back(k);

	// 균등 분배: 각 키가 동일한 개수로 S에 등장하도록 할당
	const uint32_t U = static_cast<uint32_t>(keys.size());
	if (U == 0) {
		return std::vector<Tuple>(S_LENGTH); // 비정상 상황: 빈 R. 빈 초기화 반환
	}

	const uint32_t base = S_LENGTH / U;
	uint32_t rem = S_LENGTH % U;

	static std::mt19937 rng(
		static_cast<uint32_t>(
			std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ 0x9e3779b9
		)
	);
	std::uniform_int_distribution<uint32_t> dist32(0u, 0xFFFFFFFFu);

	// remainder를 공정하게 분산하기 위해 키 순서를 셔플
	std::shuffle(keys.begin(), keys.end(), rng);

	std::vector<Tuple> S;
	S.reserve(S_LENGTH);
	for(uint32_t i = 0; i < U; ++i) {
		uint32_t count = base + (rem ? 1u : 0u);
		if (rem) rem--;
		for(uint32_t c = 0; c < count; ++c) {
			Tuple t;
			t.key = keys[i];
			t.rid = dist32(rng);
			S.push_back(t);
		}
	}

	// 최종 순서를 랜덤화
	std::shuffle(S.begin(), S.end(), rng);
	return S;
}