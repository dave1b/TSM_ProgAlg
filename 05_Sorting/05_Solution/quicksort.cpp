#include <cassert>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>
#include <execution>
#include <random>
#include "Stopwatch.h"
#include "checkresult.h"

using Vector = std::vector<float>;

////////////////////////////////////////////////////////////////////////////////////////
// compiler directives
//#define _RANDOMPIVOT_ // random pivot chosing is too slow

#ifndef _RANDOMPIVOT_
////////////////////////////////////////////////////////////////////////////////////////
// determine median of a[p1], a[p2], and a[p3]
static int median(float a[], int p1, int p2, int p3) {
	const float ap1 = a[p1];
	const float ap2 = a[p2];
	const float ap3 = a[p3];

	if (ap1 <= ap2) {
		return (ap2 <= ap3) ? p2 : ((ap1 <= ap3) ? p3 : p1);
	} else {
		return (ap1 <= ap3) ? p1 : ((ap2 <= ap3) ? p3 : p2);
	}
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
// serial quicksort
// sorts a[left]..a[right]
void quicksort(float a[], int left, int right) {
	// compute pivot
#ifdef _RANDOMPIVOT_
	std::default_random_engine e;
	std::uniform_int_distribution dist(left, right);

	const size_t pivotPos = dist(e);
#else
	const size_t pivotPos = median(a, left, left + (right - left)/2, right);
#endif
	const float pivot = a[pivotPos];

	int i = left, j = right;

	do {
		while (a[i] < pivot) i++;
		while (pivot < a[j]) j--;
		if (i <= j) {
			std::swap(a[i], a[j]);
			i++;
			j--;
		}
	} while (i <= j);
	if (left < j) quicksort(a, left, j);
	if (i < right) quicksort(a, i, right);
}

////////////////////////////////////////////////////////////////////////////////////////
// sorts a[left]..a[right] using p threads and b as a temporary storage
static void pQsort(float a[], float b[], int left, int right, int p) {
	if (left >= right) return;

	assert(a);
	assert(p > 0);

	if (p == 1) {
		quicksort(a, left, right);
		//std::sort(a + left, a + right + 1); // is slighlty faster
	} else {
		assert(p > 1);
		const int pp = p + 1;
		const int n = right - left + 1;
		const int nlocal = 1 + n/p;

		std::vector<int> s(pp); s[p] = right + 1;// start indices per thread
		std::vector<int> l(p);					// index positions of the elements larger than pivot
		std::vector<int> q(pp);					// left + prefix sums of the number of elements <= pivot
		std::vector<int> r(pp);					// q[p] + prefix sums of the number of elements >= pivot

		// compute pivot position
#ifdef _RANDOMPIVOT_
		std::default_random_engine e;
		std::uniform_int_distribution dist(left, right);

		const int pivotPos = dist(e);
#else
		const int pivotPos = median(a, left, left + (right - left)/2, right);
#endif
		const float pivot = a[pivotPos];

		#pragma omp parallel default(shared) num_threads(p)
		{
			// compute s
			#pragma omp for
			for (int k = 0; k < p; k++) {
				s[k] = std::min(right + 1, left + k*nlocal); // save start index of small elements per thread
			}

			// compute l and locally rearrange in parallel: O(n/p)
			#pragma omp for
			for (int k = 0; k < p; k++) {
				const int beg = s[k];
				const int end = s[k + 1];
				int i = beg;
				int j = end - 1;

				if (beg < end) {
					// thread has non-empty partition
					do {
						// additional checks are necessary because a partition doesn't need to contain a pivot value
						while (i < end && a[i] < pivot) i++;
						while (j >= beg && pivot < a[j]) j--;
						if (i <= j) {
							std::swap(a[i], a[j]);
							i++;
							j--;
						}
					} while (i <= j);
				}

				l[k] = i; // save start index of large elements per thread
				assert(beg <= l[k] && l[k] <= end);

#ifdef _DEBUG
				for (int ai = beg; ai < l[k]; ai++) {
					assert(a[ai] <= pivot);
				}
				for (int ai = l[k]; ai < end; ai++) {
					assert(a[ai] >= pivot);
				}
#endif
			}

			// compute prefix-sums q and r: O(p), simple but not optimal
			#pragma omp single
			{
				q[0] = left;
				for (int k = 0; k < p; k++) {
					const int nSmall = l[k] - s[k];
					assert(0 <= nSmall && nSmall <= nlocal);
					q[k + 1] = q[k] + nSmall;
				}
				r[0] = q[p];
				for (int k = 0; k < p; k++) {
					const int nLarge = s[k + 1] - l[k];
					assert(0 <= nLarge && nLarge <= nlocal);
					r[k + 1] = r[k] + nLarge;
				}
			}

			// global rearrangement

			// rearrange in parallel while copying to array b: O(n/p)
			#pragma omp for
			for (int k = 0; k < p; k++) {
				const int nSmall = l[k] - s[k];
				const int nLarge = s[k + 1] - l[k];

				for (int i = 0; i < nSmall; i++) {
					assert(a[s[k] + i] <= pivot);
					b[q[k] + i] = a[s[k] + i];
				}
				for (int j = 0; j < nLarge; j++) {
					assert(a[l[k] + j] >= pivot);
					b[r[k] + j] = a[l[k] + j];
				}
			}
		}

		// serial copy from b to a: O(n), simpler and faster than parallel copying
		memcpy(a + left, b + left, n*sizeof(float));

		// splitting position
		const int m = r[0] - 1;

#ifdef _DEBUG
		for (int i = left; i <= m; i++) {
			assert(a[i] <= pivot);
		}
		for (int i = m + 1; i <= right; i++) {
			assert(a[i] >= pivot);
		}
#endif

		// partition processes
		const int p1 = std::max(1, p*(m - left)/(right - left));

		#pragma omp parallel sections default(shared) num_threads(2)
		{
			#pragma omp section
			pQsort(a, b, left, m, p1);
			#pragma omp section
			pQsort(a, b, m + 1, right, p - p1);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////
// parallel quicksort
// sorts a[left]..a[right] using p threads 
void parallelQuicksort(float a[], int left, int right, int p) {
	assert(p > 0);
	assert(left >= 0 && left <= right);

	// TODO use OMP
	Vector b(right + 1); // temporary storage

	pQsort(a, b.data(), left, right, p);
}

////////////////////////////////////////////////////////////////////////////////////////
void quicksortTests(int n) {
	std::cout << "\nQuicksort Tests" << std::endl;
	Stopwatch sw;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;
	Vector data(n);
	Vector sortRef(n);
	Vector sort(n);

	// init arrays
	for (size_t i = 0; i < n; i++) sortRef[i] = data[i] = dist(e);
	const int p = omp_get_num_procs();

	// omp settings
	//omp_set_nested(true);
	omp_set_max_active_levels(30);
	std::cout << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "p = " << p << std::endl;
	std::cout << "Max Threads: " << omp_get_max_threads() << std::endl;
//	std::cout << "Nested Threads: " << boolalpha << (bool)omp_get_nested() << std::endl << std::endl;
	std::cout << "Nested Levels: " << omp_get_max_active_levels() << std::endl << std::endl;

	// stl sort
	sw.Start();
	std::sort(sortRef.begin(), sortRef.end());
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("std::sort:", sortRef.data(), sortRef.data(), ts, ts, n, p);

	// parallel stl sort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	std::sort(std::execution::par, sort.begin(), sort.end());
	sw.Stop();
	check("parallel sort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// sequential quicksort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	quicksort(sort.data(), 0, n - 1);
	sw.Stop();
	check("sequential quicksort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel quicksort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	parallelQuicksort(sort.data(), 0, n - 1, p);
	sw.Stop();
	check("parallel quicksort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);
}