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
static void compareSplit(int nlocal, float sent[], float received[], float result[], bool keepSmall, bool &changed)
{
	changed = false;
	if (keepSmall)
	{
		for (int i = 0, j = 0, k = 0; k < nlocal; k++)
		{
			if (j == nlocal || (i < nlocal && sent[i] <= received[j]))
			{
				result[k] = sent[i++];
			}
			else
			{
				result[k] = received[j++];
				changed = true;
			}
		}
	}
	else
	{
		const int last = nlocal - 1;
		for (int i = last, j = last, k = last; k >= 0; k--)
		{
			if (j == -1 || (i >= 0 && sent[i] >= received[j]))
			{
				result[k] = sent[i--];
			}
			else
			{
				result[k] = received[j--];
				changed = true;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// elements to be sorted
// received and temp are auxiliary buffers
static void oddEvenSort(const int p, const int nlocal, const int id, float elements[], float received[], float temp[])
{
	// TODO use MPI
	int idOdd, idEven;
	MPI_Status status;
	bool changedOdd = true, changedEven = true;

	// determine the id of the processors that id needs to communicate during the odd and even phases
	if (id & 1)
	{
		idOdd = id + 1;
		idEven = id - 1;
	}
	else
	{
		idOdd = id - 1;
		idEven = id + 1;
	}
	if (idEven < 0 || idEven == p)
		idEven = MPI_PROC_NULL;
	if (idOdd < 0 || idOdd == p)
		idOdd = MPI_PROC_NULL;

	// main loop of odd-even sort: local data to send is in elements buffer
	// loop until every process keeps its own data for two successive iterations
	for (int i = 0; i < p && (changedOdd || changedEven); i++)
	{
		bool lChanged = false;

		if (i & 1)
		{
			// odd phase
			MPI_Sendrecv(elements, nlocal, MPI_FLOAT, idOdd, 1, received, nlocal, MPI_FLOAT, idOdd, 1, MPI_COMM_WORLD, &status);
		}
		else
		{
			// even phase
			MPI_Sendrecv(elements, nlocal, MPI_FLOAT, idEven, 1, received, nlocal, MPI_FLOAT, idEven, 1, MPI_COMM_WORLD, &status);
		}
		if (status.MPI_SOURCE != MPI_PROC_NULL)
		{
			// sent data is in elements buffer
			// received data is in received buffer
			compareSplit(nlocal, elements, received, temp, id < status.MPI_SOURCE, lChanged);
			// temp contains result of compare-split operation: copy temp back to elements buffer
			std::copy(temp, temp + nlocal, elements);
		}

		// reduce changed value: all processes have the same information in changedOdd or changedEven
		if (i & 1)
		{
			MPI_Allreduce(&lChanged, &changedOdd, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Allreduce(&lChanged, &changedEven, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
		}
		// std::cout << i << std::endl;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// 1. phase: compare-split over long distances in log(p) steps
// 2. phase: odd-even-sort
// nProcs: number of processes
// nlocal: number of elements to be sorted
// elements: partioned data to be sorted
void shellSort(const int nProcs, const int nlocal, const int myID, float elements[])
{
	// TODO use MPI
	MPI_Status status;
	std::vector<float> received(nlocal);
	std::vector<float> temp(nlocal);
	bool changed = true;

	// sort local elements
	std::sort(elements, elements + nlocal);

	// nProcs doesn't need to be a power of two
	int p = nProcs - 1;
	int p2 = 1;
	while (p)
	{
		p2 <<= 1;
		p >>= 1;
	}
	assert(p2 >= nProcs);

	// phase 1: p2 must be a power of two
	int groupSize = p2;

	while (groupSize > 1)
	{
		bool lChanged = false;
		div_t qr = div(myID, groupSize); // group = qr.quot, local id = qr.rem
		int partner = (qr.quot + 1) * groupSize - 1 - qr.rem;
		if (partner >= nProcs)
			partner = MPI_PROC_NULL;

		// exchange data with partner
		MPI_Sendrecv(elements, nlocal, MPI_FLOAT, partner, 1, received.data(), nlocal, MPI_FLOAT, partner, 1, MPI_COMM_WORLD, &status);
		if (partner != MPI_PROC_NULL)
		{
			compareSplit(nlocal, elements, received.data(), temp.data(), qr.rem < groupSize / 2, lChanged);
			// copy from temp to elements
			copy(temp.begin(), temp.end(), elements);
		}

		// reduce changed value: all processes have the same information in changed
		// MPI_Allreduce(&lChanged, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD); // doesn't influence the parallel run time

		// update groupSize
		groupSize >>= 1;
	}

	// phase 2: data is in elements buffer
	if (changed)
		oddEvenSort(nProcs, nlocal, myID, elements, received.data(), temp.data());
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void shellSortTests()
{
	int p, id;
	double seqElapsed = 0;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	if (id == 0)
	{
		std::cout << "Shellsort with " << p << " MPI processes" << std::endl;
	}

	for (int i = 15; i <= 27; i += 3)
	{
		const int n = 1 << i;
		const int nlocal = n / p;
		assert(nlocal * p == n);

		std::vector<float> sorted;
		std::vector<float> elements;
		std::vector<float> received(nlocal);

		if (id == 0)
		{
			sorted.resize(n);

			// fill in elements with random numbers
			elements.resize(n);
			for (int j = 0; j < n; j++)
			{
				elements[j] = sorted[j] = (float)rand();
			}

			// make copy and sort the copy with std::sort
			double seqStart = MPI_Wtime();
			std::sort(sorted.begin(), sorted.end());
			seqElapsed = MPI_Wtime() - seqStart;
		}
		else
		{
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
		if (id == 0)
		{
			if (sorted == elements)
			{
				std::cout << n << " elements have been sorted in ascending order in " << elapsed << " s" << std::endl;
				std::cout << p << " processes" << std::endl;
				std::cout << "speedup S = " << seqElapsed / elapsed << std::endl;
			}
			else
			{
				std::cout << "elements are not correctly sorted" << std::endl;
			}
		}
	}
}