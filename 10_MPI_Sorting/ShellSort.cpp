#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstring>
#include <random>
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// compare-split of nlocal data elements
// pre-condition:  sent contains the data sent to another process
//                 received contains the data received by the other process
// post-condition: result contains the kept data elements
//                 changed is true if result contains received data
static void compareSplit(int nlocal, float *sent, float *received, float *result, bool keepSmall, bool& changed) {
	changed = false;
	if (keepSmall) {
		for (int i = 0, j = 0, k = 0; k < nlocal; k++) {
			if (j == nlocal || (i < nlocal && sent[i] <= received[j])) {
				result[k] = sent[i++];
			} else {
				result[k] = received[j++];
				changed = true;
			}
		}
	} else {
		const int last = nlocal - 1;
		for (int i = last, j = last, k = last; k >= 0; k--) {
			if (j == -1 || (i >= 0 && sent[i] >= received[j])) {
				result[k] = sent[i--];
			} else {
				result[k] = received[j--];
				changed = true;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// elements to be sorted
// received and temp are auxiliary buffers
static void oddEvenSort(const int p, const int nlocal, const int id, float* elements, float* received, float* temp) {
	// TODO use MPI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// 1. phase: compare-split over long distances in log(p) steps
// 2. phase: odd-even-sort
// nProcs: number of processes
// nlocal: number of elements to be sorted
// elements: partioned data to be sorted
void shellSort(const int nProcs, const int nlocal, const int myID, float* elements) {
	// TODO use MPI
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void shellsort() {
	int p, id;
	double seqElapsed = 0;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	if (id == 0) {
		std::cout << "Shellsort with " << p << " MPI processes" << std::endl;
	}

	for (int i = 15; i <= 27; i += 3) {
		const int n = 1 << i;
		const int nlocal = n/p;
		assert(nlocal*p == n);

		std::vector<float> sorted;
		std::vector<float> elements;
		std::vector<float> received(nlocal);

		if (id == 0) {
			sorted.resize(n);

			// fill in elements with random numbers
			elements.resize(n);
			for (int j = 0; j < n; j++) {
				elements[j] = sorted[j] = (float)rand();
			}

			// make copy and sort the copy with std::sort
			double seqStart = MPI_Wtime();
			std::sort(sorted.begin(), sorted.end());
			seqElapsed = MPI_Wtime() - seqStart;

		} else {
			elements.resize(nlocal);
		}

		// send partitioned elements to processes
		MPI_Scatter(elements.data(), nlocal, MPI_FLOAT, received.data(), nlocal, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// use a barrier to synchronize start time
		MPI_Barrier(MPI_COMM_WORLD);
		const double start = MPI_Wtime();

		shellSort(p, nlocal, id, received.data());

		// stop time
		double localElapsed = MPI_Wtime() - start, elapsed;

		// reduce maximum time
		MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		// send sorted local elements to process 0
		MPI_Gather(received.data(), nlocal, MPI_FLOAT, elements.data(), nlocal, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// check if all elements are sorted in ascending order
		if (id == 0) {
			if (sorted == elements) {
				std::cout << n << " elements have been sorted in ascending order in " << elapsed << " s" << std::endl;
				std::cout << p << " processes" << std::endl;
				std::cout << "speedup S = " << seqElapsed/elapsed << std::endl;
			} else {
				std::cout << "elements are not correctly sorted" << std::endl;
			}
		}
	}
}