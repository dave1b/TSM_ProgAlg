#include <cassert>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <omp.h>
#include <Stopwatch.h>
#include "checkresult.h"

using Vector = std::vector<float>;

///////////////////////////////////////////////////////////////////////////////
// Sequential Bitonic sort (used in performance tests)
// n must be a power of 2
static void bitonicSortSeq(float a[], const int n) {
	// compute d = log(n)
	int nn = n;
	int d = -1;

	while (nn) {
		d++;
		nn >>= 1;
	}

	int biti = 1;
	for (int i = 0; i < d; i++) {
		int bitj = biti; // bit j

		biti <<= 1; // bit i + 1
		for (int j = i; j >= 0; j--) {
			for (int k = 0; k < n; k++) {
				const int m = (~k & bitj) | (k & ~bitj); // xor

				if (m > k) {
					// only one of the two processing elements initiates the swap operation
					const bool bi = (k & biti) != 0;
					const bool bj = (k & bitj) != 0;
					if (bi == bj) {
						// comp_exchange_min on channel k with m
						// k takes the min, m the max
						if (a[k] > a[m]) std::swap(a[k], a[m]);
					} else {
						// comp_exchange_max on channel k with m
						// k takes the max, m the min
						if (a[k] < a[m]) std::swap(a[k], a[m]);
					}
				}
			}
			bitj >>= 1;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Bitonic sort implementation for p = n
// n must be a power of 2
// p parallel threads
static void bitonicSortOMP1(float a[], const int n, const int p) {
	// TODO use OMP
	int nn = n;
	int d = -1;

	while (nn) {
		d++;
		nn >>= 1;
	}

	int biti = 1;

	#pragma omp parallel default(none) firstprivate(biti) shared(a,d,n) //num_threads(n)
	for (int i = 0; i < d; i++) {
		int bitj = biti; // bit j

		biti <<= 1; // bit i + 1
		for (int j = i; j >= 0; j--) {

			#pragma omp for 
			for (int k = 0; k < n; k++) {
				//std::cout << omp_get_num_threads() << std::endl;
				const int m = (~k & bitj) | (k & ~bitj); // xor

				if (m > k) {
					// only one of the two processing elements initiates the swap operation
					const bool bi = (k & biti) != 0;
					const bool bj = (k & bitj) != 0;
					if (bi == bj) {
						// comp_exchange_min on channel k with m
						// k takes the min, m the max
						if (a[k] > a[m]) std::swap(a[k], a[m]);
					} else {
						// comp_exchange_max on channel k with m
						// k takes the max, m the min
						if (a[k] < a[m]) std::swap(a[k], a[m]);
					}
				}
			} // implicit barrier

			bitj >>= 1;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// compare-split of nlocal data elements
// input: a and b
// output: small and large
static void compareSplit(int nlocal, float a[], float b[], float small[], float large[]) {
	// TODO (use OMP sections)
	const int last = nlocal - 1;

	#pragma omp parallel sections
	{
		#pragma omp section
		for (int i = 0, j = 0, k = 0; k < nlocal; k++) {
			if (j == nlocal || (i < nlocal && a[i] <= b[j])) {
				small[k] = a[i++];
			} else {
				small[k] = b[j++];
			}
		}
		#pragma omp section
		for (int i = last, j = last, k = last; k >= 0; k--) {
			if (j == -1 || (i >= 0 && a[i] >= b[j])) {
				large[k] = a[i--];
			} else {
				large[k] = b[j--];
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Bitonic sort implementation for p < n, O(log^2(n))
// (used in performance tests)
// n and p must be a power of 2
// p parallel threads
void bitonicSortOMP2(float a[], const int n, const int p) {
	// TODO use OMP
	const int nlocal = n/p;
	assert(nlocal*p == n);

	Vector tmp(n);

	// local sort
	#pragma omp parallel for default(shared) num_threads(p)
	for (int i = 0; i < p; i++) {
		std::sort(a + i*nlocal, a + (i + 1)*nlocal);
	}

	int nn = p;
	int d = -1;
	while (nn) {
		d++;
		nn >>= 1;
	}

	int biti = 1;
	for (int i = 0; i < d; i++) {
		int bitj = biti; // bit j
		biti <<= 1; // bit i + 1
		for (int j = i; j >= 0; j--) {

			#pragma omp parallel for default(shared) num_threads(p)
			for (int k = 0; k < p; k++) {
				const int m = (~k & bitj) | (k & ~bitj); // xor

				if (m > k) {
					// only one of the two processing elements initiates the compare-split operation
					float* const ak = a + k*nlocal;
					float* const am = a + m*nlocal;
					float* const tk = tmp.data() + k*nlocal;
					float* const tm = tmp.data() + m*nlocal;
					const bool bi = (k & biti) != 0;
					const bool bj = (k & bitj) != 0;
					if (bi == bj) {
						// comp_split_min on channel k with m
						// k takes the min, m the max
						compareSplit(nlocal, ak, am, tk, tm);
					} else {
						// comp_split_max on channel k with m
						// k takes the max, m the min
						compareSplit(nlocal, ak, am, tm, tk);
					}
					std::copy(tk, tk + nlocal, ak);
					std::copy(tm, tm + nlocal, am);
				}
			}

			bitj >>= 1;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////
void bitonicsortTests(int n) {
	std::cout << "\nBitonic Sort Tests" << std::endl;
	Stopwatch sw;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;
	Vector data(n);
	Vector sortRef(n);
	Vector sort(n);

	//omp_set_nested(true);
	omp_set_max_active_levels(2);

	// init arrays
	for (int i = 0; i < n; i++) sortRef[i] = data[i] = dist(e);
	int p = omp_get_num_procs();

	// omp settings
	std::cout << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "p = " << p << std::endl;
	std::cout << "Max Threads: " << omp_get_max_threads() << std::endl;

	// stl sort
	sw.Start();
	std::sort(sortRef.begin(), sortRef.end());
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("std::sort:", sortRef.data(), sortRef.data(), ts, ts, n, p);

	// sequential bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	bitonicSortSeq(sort.data(), n);
	sw.Stop();
	check("sequential bitonic sort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	bitonicSortOMP1(sort.data(), n, p);
	sw.Stop();
	check("parallel bitonic sort (p = n):", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	p = 8; assert(n%p == 0);
	sw.Restart();
	bitonicSortOMP2(sort.data(), n, p);
	sw.Stop();
	check("parallel bitonic sort (p < n):", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);
}
