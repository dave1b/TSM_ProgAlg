#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include "Stopwatch.h"
#include "checkresult.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// Explicit computation
static int64_t sum(const int64_t n) {
	return n*(n + 1)/2;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Sequential summation
static int64_t sumSerial(const std::vector<int>& arr) {
	int64_t sum = 0;

	for (auto& v : arr) {
		sum += v;
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation using parallel for_each in C++20 and atomic_int64_t
static int64_t sumPar1(const std::vector<int>& arr) {
	// TODO: use std::atomic_int64_t and std::for_each with std::execution::par
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation using parallel reduce in C++20 and implicit reduction
static int64_t sumPar2(const std::vector<int>& arr) {
	// TODO use std::reduce
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation using parallel reduce in C++20 and explicit reduction
static int64_t sumPar3(const std::vector<int>& arr) {
	// TODO use std::reduce and lambda expression [](int64_t a, int64_t b) {... }
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Different summation tests
void summation() {
	std::cout << "\nSummation Tests" << std::endl;

	Stopwatch sw;
	std::vector<int> arr(10'000'000);

	std::iota(arr.begin(), arr.end(), 1);

	sw.Start();
	const int64_t sum0 = sum((int64_t)arr.size());
	sw.Stop();
	check("Explicit:", sum0, sum0, sw.GetElapsedTimeMilliseconds(), sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const int64_t sumS = sumSerial(arr);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("Sequential:", sum0, sumS, ts, ts);

	sw.Restart();
	const int64_t sum7 = sumPar1(arr);
	sw.Stop();
	check("Parallel for_each Atomic int:", sum0, sum7, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const int64_t sum8 = sumPar2(arr);
	sw.Stop();
	check("Parallel implicit reduction:", sum0, sum8, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const int64_t sum9 = sumPar3(arr);
	sw.Stop();
	check("Parallel explicit reduction:", sum0, sum9, ts, sw.GetElapsedTimeMilliseconds());
}