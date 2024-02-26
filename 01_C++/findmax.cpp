#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>
#include <vector>
#include <iomanip>
#include <thread>
#include <random>
#include "Stopwatch.h"
#include "checkresult.h"

static const unsigned p = std::thread::hardware_concurrency();

//////////////////////////////////////////////////////////////////////////////////////////////
// Sequential search
static double findSerial(const std::vector<double> &arr)
{
	return *std::max_element(arr.begin(), arr.end());
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel search with max_element
static double findPar1(const std::vector<double> &arr)
{
	// TODO use std::max_element
	// std::atomic_int64_t sum{0};

	// std::int64_t num_of_items = arr.size();
	// std::int64_t steps_size = num_of_items / p;
	// std::cout << steps_size;

	return *std::max_element(std::execution::par, arr.begin(), arr.end());
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel reduction
static double findPar2(const std::vector<double> &arr)
{
	// TODO use std::reduce with a lambda expression
	return *std::max_element(std::execution::par, arr.begin(), arr.end(),
							 [](double a, double b)
							 {
								 return std::max(a, b);
							 });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Different search tests
void findMaximum()
{
	std::cout << "\nFind Maximum Tests" << std::endl;
	std::cout << "Processing units " << p << std::endl;

	Stopwatch sw;
	std::vector<double> arr(10'000'000);
	std::default_random_engine e;
	std::uniform_real_distribution dist;

	for (size_t i = 0; i < arr.size(); i++)
		arr[i] = dist(e);

	sw.Start();
	const double maxS = findSerial(arr);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("Sequential:", maxS, maxS, ts, ts);

	sw.Restart();
	const double max1 = findPar1(arr);
	sw.Stop();
	check("Parallel max_element:", maxS, max1, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	const double max2 = findPar2(arr);
	sw.Stop();
	check("Parallel reduction:", maxS, max2, ts, sw.GetElapsedTimeMilliseconds());
}
