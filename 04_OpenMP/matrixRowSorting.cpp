#include <iostream>
#include <cmath>
#include <climits>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <random>
#include "Stopwatch.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// sequentially sorting all arrays A[i]
void matrixSortSeq(std::vector<std::vector<int>> &A)
{
	for (size_t i = 0; i < A.size(); i++)
	{
		std::sort(A[i].begin(), A[i].end());
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// parallel sorting all arrays A[i]
void matrixSortOmp(std::vector<std::vector<int>> &A)
{
	// TODO use OMP to parallelize a for loop
	int max_threads = omp_get_max_threads();
	omp_set_num_threads(max_threads);
	int n_threads = omp_get_num_threads();
	int size = A.size();
	int chunk = size / n_threads;
	std::cout << "Number of threads: " << n_threads << std::endl;
	std::cout << "Max number of threads: " << max_threads << std::endl;
#pragma omp parallel for shared(A) schedule(static, chunk) num_threads(max_threads)
	for (size_t i = 0; i < A.size(); i++)
	{
		std::sort(A[i].begin(), A[i].end());
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template <typename T>
static void check(const char text[], const T &ref, const T &result, double ts, double tp)
{
	static const int p = omp_get_num_procs();
	const double S = ts / tp;
	const double E = S / p;

	std::cout << std::setw(30) << std::left << text << result.size();
	std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << ", E = " << E << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl
			  << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void matrixRowSortingTests()
{
	std::cout << "\nMatrix Row Sorting Tests" << std::endl;

	Stopwatch swSER, swOMP;
	std::default_random_engine e;
	std::uniform_int_distribution dist;

	for (size_t n1 = 1000; n1 <= 2000; n1 += 200)
	{
		std::cout << "n = " << n1 << std::endl;
		std::vector<std::vector<int>> A(n1);
		std::vector<std::vector<int>> B(n1);

		for (size_t i = 0; i < n1; i++)
		{
			A[i].resize(50000);
			B[i].resize(50000);

			for (size_t j = 0; j < A[i].size(); j++)
			{
				A[i][j] = B[i][j] = dist(e);
			}
		}

		// run serial implementation
		swSER.Restart();
		matrixSortSeq(A);
		swSER.Stop();
		const double ts = swSER.GetElapsedTimeMilliseconds();

		// run OMP matrix sorting
		swOMP.Restart();
		matrixSortOmp(B);
		swOMP.Stop();

		check("Matrix Row Sorting:", A, B, ts, swOMP.GetElapsedTimeMilliseconds());
	}
}
