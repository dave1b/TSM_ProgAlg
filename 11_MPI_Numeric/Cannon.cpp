#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include "mpi.h"
#include "Stopwatch.h"
#include "checkresult.h"

void matMultSeqStandard(const int a[], const int b[], int c[], const int n);
void matMultSeq(const int a[], const int b[], int c[], const int n);

using Vector = std::vector<int>;

//////////////////////////////////////////////////////////////////////////////////////////////
// Cannon's algorithm using blocking send and receive operations and Cartesian grid
static void cannonBlocking(int a[], int b[], int c[], const int nlocal, const int pSqrt)
{
	const int dims[] = {pSqrt, pSqrt}; // [y,x]
	const int periods[] = {true, true};
	MPI_Comm comm2D;

	// set up the Cartesian topology with wraparound connections and rank reordering
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2D);

	// get rank and coordinates with respect to the new topology
	int my2Drank;
	int myCoords[2]; // [y,x]

	MPI_Comm_rank(comm2D, &my2Drank);
	MPI_Cart_coords(comm2D, my2Drank, 2, myCoords);

	// initialize C
	const int size = nlocal * nlocal;
	std::fill_n(c, size, 0);

	// perform the initial matrix alignment: first for A then for B
	int shiftSrc, shiftDst;

	MPI_Cart_shift(comm2D, 1, -myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(a, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, -myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(b, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// compute ranks of the left and up shifts
	int leftRank, rightRank, downRank, upRank;

	MPI_Cart_shift(comm2D, 1, -1, &rightRank, &leftRank);
	MPI_Cart_shift(comm2D, 0, -1, &downRank, &upRank);

	// main computation loop
	for (int i = 0; i < pSqrt; i++)
	{
		// matrix multiplication: cLocal += aLocal * bLocal
		matMultSeq(a, b, c, nlocal);

		// shift A left by one
		MPI_Sendrecv_replace(a, size, MPI_INT, leftRank, 1, rightRank, 1, comm2D, MPI_STATUSES_IGNORE);

		// shift B up by one
		MPI_Sendrecv_replace(b, size, MPI_INT, upRank, 1, downRank, 1, comm2D, MPI_STATUSES_IGNORE);
	}

	// restore the original distribution of A and B
	MPI_Cart_shift(comm2D, 1, myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(a, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(b, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// free up communicator
	MPI_Comm_free(&comm2D);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Cannon's algorithm using non-blocking send and receive operations and Cartesian grid
// nlocal is the local number of elements
static void cannonNonBlocking(int a[], int b[], int c[], const int nlocal, const int pSqrt)
{
	// TODO use MPI
	// set up the Cartesian topology with wrapparound connections and rank reordering
	const int dims[] = {pSqrt, pSqrt}; // [y,x]
	const int periods[] = {true, true};
	MPI_Comm comm2D;

	// set up the Cartesian topology with wraparound connections and rank reordering
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2D);

	// get rank and coordinates with respect to the new topology
	int my2Drank;
	int myCoords[2]; // [y,x]

	MPI_Comm_rank(comm2D, &my2Drank);
	MPI_Cart_coords(comm2D, my2Drank, 2, myCoords);

	// create local buffers
	const int size = nlocal * nlocal;
	std::vector<int> a1(size);
	std::vector<int> b1(size);
	int *aBuf[] = {a, a1.data()};
	int *bBuf[] = {b, b1.data()};

	// initialize C
	std::fill_n(c, size, 0);

	// perform the initial matrix alignment: first for A then for B
	int shiftSrc, shiftDst;

	MPI_Cart_shift(comm2D, 1, -myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(a, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, -myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(b, size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// compute ranks of the left and up shifts
	int leftRank, rightRank, downRank, upRank;
	const int nRequests = 4;
	MPI_Request requests[nRequests];

	MPI_Cart_shift(comm2D, 1, -1, &rightRank, &leftRank);
	MPI_Cart_shift(comm2D, 0, -1, &downRank, &upRank);

	// main computation loop
	int i;
	for (i = 0; i < pSqrt; i++)
	{
		// initiate non-blocking shifting
		MPI_Isend(aBuf[i & 1], size, MPI_INT, leftRank, 1, comm2D, requests);
		MPI_Isend(bBuf[i & 1], size, MPI_INT, upRank, 1, comm2D, requests + 1);
		MPI_Irecv(aBuf[(i + 1) & 1], size, MPI_INT, rightRank, 1, comm2D, requests + 2);
		MPI_Irecv(bBuf[(i + 1) & 1], size, MPI_INT, downRank, 1, comm2D, requests + 3);

		// matrix multiplication: cLocal += aLocal * bLocal
		matMultSeq(aBuf[i & 1], bBuf[i & 1], c, nlocal);

		// wait for shifted data
		MPI_Waitall(nRequests, requests, MPI_STATUSES_IGNORE);
	}

	// restore the original distribution of A and B
	MPI_Cart_shift(comm2D, 1, myCoords[0], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(aBuf[i & 1], size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);
	MPI_Cart_shift(comm2D, 0, myCoords[1], &shiftSrc, &shiftDst);
	MPI_Sendrecv_replace(bBuf[i & 1], size, MPI_INT, shiftDst, 1, shiftSrc, 1, comm2D, MPI_STATUSES_IGNORE);

	// free up communicator
	MPI_Comm_free(&comm2D);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Cannon's matrix multiplication test program
void cannonsTests()
{
	constexpr bool verbose = true;
	int wrongMPIresults = 0;
	int p, id;
	bool blocking = false;
	Stopwatch swCPU, swPAR, swMPI;
	std::default_random_engine e;

	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	const int pSqrt = (int)sqrt(p);

	if (pSqrt * pSqrt != p)
	{
		if (id == 0)
			std::cerr << "number of processes " << p << " must be a square number" << std::endl;
		return;
	}
	else
	{
		if (id == 0)
		{
			std::cout << std::endl
					  << "Cannon's matrix multiplication" << std::endl;

			std::cout << "Blocking [true/false] ";
			std::cin >> std::boolalpha >> blocking;
		}
		// send blocking to all processes
		MPI_Bcast(&blocking, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

		if (blocking)
		{
			std::cout << "Cannon's blocking algorithm started" << std::endl;
		}
		else
		{
			std::cout << "Cannon's non-blocking algorithm started" << std::endl;
		}
	}

	for (int n = 1000; n <= 2000; n += 200)
	{
		const int nLocal = n / pSqrt;
		const int nLocal2 = nLocal * nLocal;
		const int maxVal = (int)sqrt(INT_MAX / n);

		std::uniform_int_distribution dist(0, maxVal);
		Vector aLocal(nLocal2);
		Vector bLocal(nLocal2);
		Vector cLocal(nLocal2);

		if (id == 0)
		{
			const int n1 = nLocal * pSqrt;
			const int n2 = n1 * n1;

			Vector A(n2);	 // matrix A
			Vector B(n2);	 // matrix B
			Vector C(n2);	 // matrix C (reference)
			Vector Cpar(n2); // matrix C (parallel)
			Vector tmp(n2);

			for (int i = 0; i < n2; i++)
			{
				A[i] = dist(e);
				B[i] = dist(e);
			}

			// run reference implementation: compute C
			swCPU.Restart();
			// matMultSeqStandard(A.data(), B.data(), C.data(), n1);
			matMultSeq(A.data(), B.data(), C.data(), n1);
			swCPU.Stop();
			const double ts = swCPU.GetElapsedTimeMilliseconds();
			std::cout << "Serial on CPU in " << ts << " ms" << std::endl;

			swMPI.Restart();
			int *t = tmp.data();

			// partition and distribute matrix A
			for (int i = 0; i < pSqrt; i++)
			{
				for (int j = 0; j < pSqrt; j++)
				{
					int *arow = A.data() + i * nLocal * n1 + j * nLocal;

					// copy block a(ij) to array tmp
					for (int k = 0; k < nLocal; k++)
					{
						std::copy(arow, arow + nLocal, t);
						t += nLocal;
						arow += n1;
					}
				}
			}
			MPI_Scatter(tmp.data(), nLocal2, MPI_INT, aLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// partition and distribute matrix B
			t = tmp.data();
			for (int i = 0; i < pSqrt; i++)
			{
				for (int j = 0; j < pSqrt; j++)
				{
					int *brow = B.data() + i * nLocal * n1 + j * nLocal;

					// copy block b(ij) to array tmp
					for (int k = 0; k < nLocal; k++)
					{
						std::copy(brow, brow + nLocal, t);
						t += nLocal;
						brow += n1;
					}
				}
			}
			MPI_Scatter(tmp.data(), nLocal2, MPI_INT, bLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication: compute Ccannon
			Vector Ccannon(n2); // matrix Ccannon

			if (blocking)
				cannonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			else
				cannonNonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			const double splitTime = swMPI.GetSplitTimeMilliseconds();

			// gather matrix C
			MPI_Gather(cLocal.data(), nLocal2, MPI_INT, tmp.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			t = tmp.data();
			for (int i = 0; i < pSqrt; i++)
			{
				for (int j = 0; j < pSqrt; j++)
				{
					int *crow = Ccannon.data() + i * nLocal * n1 + j * nLocal;

					// copy array tmp to block C1(ij)
					for (int k = 0; k < nLocal; k++)
					{
						std::copy(t, t + nLocal, crow);
						t += nLocal;
						crow += n1;
					}
				}
			}

			swMPI.Stop();
			check("Cannon:", C, Ccannon, ts, swMPI.GetElapsedTimeMilliseconds(), verbose);
			std::cout << "n = " << n << ", Cannon's split time = " << splitTime << " ms" << std::endl
					  << std::endl;
		}
		else
		{
			MPI_Scatter(nullptr, nLocal2, MPI_INT, aLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatter(nullptr, nLocal2, MPI_INT, bLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication
			if (blocking)
				cannonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			else
				cannonNonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);

			MPI_Gather(cLocal.data(), nLocal2, MPI_INT, nullptr, nLocal2, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
}