#include <execution>
#include <algorithm>
#include <iostream>
#include <future>
#include <functional>
#include <iomanip>
#include <omp.h>
#include "Stopwatch.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// Explicit computation
static int64_t sum(const int64_t n)
{
	return n * (n + 1) / 2;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Sequential summation
static int64_t sumSerial(const std::vector<int> &arr)
{
	int64_t sum = 0;

	for (auto &v : arr)
	{
		sum += v;
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation with critical section
static int64_t sumPar1(const std::vector<int> &arr)
{
	int64_t sum = 0;
	for (auto &v : arr)
	{
#pragma omp critical
		{
			sum += v;
		}
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation with explicit locks
static int64_t sumPar2(const std::vector<int> &arr)
{
	// TODO use OMP for loop parallelization and an OMP lock
	int64_t sum = 0;
	omp_lock_t write_lock;
	omp_init_lock(&write_lock);
#pragma omp parallel for schedule(static)
	for (auto &v : arr)
	{
		omp_set_lock(&write_lock);
		sum += v;
		omp_unset_lock(&write_lock);
	}
	omp_destroy_lock(&write_lock);
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel summation with OMP for loop parallelization
static int64_t sumPar3(const std::vector<int> &arr)
{
	int64_t sum = 0;
#pragma omp parallel default(none) shared(sum)
#pragma omp for schedule(static)
	for (auto &v : arr)
	{
		sum += v;
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template <typename T>
static void check(const char text[], const T &ref, const T &result, double ts, double tp)
{
	static const int p = omp_get_num_procs();
	const double S = ts / tp;
	const double E = S / p;

	std::cout << std::setw(30) << std::left << text << result;
	std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << ", E = " << E << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl
			  << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Different summation tests
void summation()
{
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
	const int64_t sum1 = sumPar1(arr);
	sw.Stop();
	check("OpenMP Critical section:", sum0, sum1, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const int64_t sum2 = sumPar2(arr);
	sw.Stop();
	check("OpenMP Explicit locks:", sum0, sum2, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const int64_t sum3 = sumPar3(arr);
	sw.Stop();
	check("OpenMP ... +=:", sum0, sum3, ts, sw.GetElapsedTimeMilliseconds());
}
