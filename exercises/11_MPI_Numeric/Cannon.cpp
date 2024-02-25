#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include "Stopwatch.h"
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// Cache aware serial implementation
// Matrix C has to be initialized with 0 in advance
static void matMultSeq(const int* a, const int* b, int* const c, const int n) {
	int* crow = c;

	for (int i = 0; i < n; i++) {
		int bpos = 0;

		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				crow[j] += a[k]*b[bpos++];
			}
		}
		a += n;
		crow += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel matrix multiplication
static void matMultPar(const int* a, const int* b, int* const c, const int n) {
	// TODO use OMP
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Cannon's algorithm using blocking send and receive operations and Cartesian grid
static void cannonBlocking(int* const a, int* const b, int* const c, const int nlocal, const int pSqrt) {
	// set up the Cartesian topology with wrapparound connections and rank reordering
	int dims[] = { pSqrt, pSqrt };	// [y,x]
	int periods[] = { true, true };
	MPI_Comm comm2D;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2D);

	// get rank and coordinates with respect to the new topology
	int my2Drank;
	int myCoords[2];	// [y,x]

	MPI_Comm_rank(comm2D, &my2Drank);
	MPI_Cart_coords(comm2D, my2Drank, 2, myCoords);

	// initialize C
	const int size = nlocal*nlocal;
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
	for (int i = 0; i < pSqrt; i++) {
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
static void cannonNonBlocking(int* const a, int* const b, int* const c, const int nlocal, const int pSqrt) {
	// TODO use MPI
}

//////////////////////////////////////////////////////////////////////////////////////////////
void matrixmultiplication() {
	int wrongMPIresults = 0;
	int p, id;
	bool blocking = false;
	Stopwatch swCPU, swPAR, swMPI;
	std::default_random_engine e;

	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	// set up the Cartesian topology with wrapparound connections and rank reordering
	int pSqrt = (int)sqrt(p);

	if (pSqrt*pSqrt != p) {
		if (id == 0) std::cerr << "number of processes " << p << " must be a square number" << std::endl;
		return;
	} else {
		if (id == 0) {
			std::cout << "Cannon's matrix multiplication" << std::endl;

			std::cout << "Blocking [true/false] ";
			std::cin >> std::boolalpha >> blocking;
		}
		// send blocking to all processes 
		MPI_Bcast(&blocking, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

		if (blocking) {
			std::cout << "Cannon's blocking algorithm started" << std::endl;
		} else {
			std::cout << "Cannon's non-blocking algorithm started" << std::endl;
		}
	}

	for (int n = 1000; n <= 2000; n += 200) {
		const int nLocal = n/pSqrt;
		const int nLocal2 = nLocal*nLocal;
		const int maxVal = (int)sqrt(INT_MAX/n);

		std::uniform_int_distribution dist(0, maxVal);
		std::vector<int> aLocal(nLocal2);
		std::vector<int> bLocal(nLocal2);
		std::vector<int> cLocal(nLocal2);

		if (id == 0) {
			const int n1 = nLocal*pSqrt;
			const int n2 = n1*n1;

			std::vector<int> A(n2);		// matrix A
			std::vector<int> B(n2);		// matrix B
			std::vector<int> C(n2);		// matrix C (reference)
			std::vector<int> Cpar(n2);	// matrix C (parallel)
			std::vector<int> tmp(n2);

			for (int i = 0; i < n2; i++) {
				A[i] = dist(e);
				B[i] = dist(e);
			}

			// run serial implementation: compute C
			swCPU.Start();
			matMultSeq(A.data(), B.data(), C.data(), n1);
			swCPU.Stop();

			// run parallel implementation (OpenMP)
			swPAR.Start();
			matMultPar(A.data(), B.data(), Cpar.data(), n1);
			swPAR.Stop();

			swMPI.Start();
			// partition and distribute matrix A
			int* t = tmp.data();

			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int* arow = A.data() + i*nLocal*n1 + j*nLocal;

					// copy block a(ij) to array tmp
					for (int k = 0; k < nLocal; k++) {
						std::copy(arow, arow + nLocal, t);
						t += nLocal;
						arow += n1;
					}
				}
			}
			MPI_Scatter(tmp.data(), nLocal2, MPI_INT, aLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// partition and distribute matrix B
			t = tmp.data();
			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int* brow = B.data() + i*nLocal*n1 + j*nLocal;

					// copy block b(ij) to array tmp
					for (int k = 0; k < nLocal; k++) {
						std::copy(brow, brow + nLocal, t);
						t += nLocal;
						brow += n1;
					}
				}
			}
			MPI_Scatter(tmp.data(), nLocal2, MPI_INT, bLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication: compute C
			std::vector<int> Ccannon(n2);	// matrix C (result Cannon)

			if (blocking) cannonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			else cannonNonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			const double splitTime = swMPI.GetSplitTimeMilliseconds();

			// gather matrix C
			MPI_Gather(cLocal.data(), nLocal2, MPI_INT, tmp.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			t = tmp.data();
			for (int i = 0; i < pSqrt; i++) {
				for (int j = 0; j < pSqrt; j++) {
					int* crow = Ccannon.data() + i*nLocal*n1 + j*nLocal;

					// copy array tmp to block C1(ij)
					for (int k = 0; k < nLocal; k++) {
						std::copy(t, t + nLocal, crow);
						t += nLocal;
						crow += n1;
					}
				}
			}

			swMPI.Stop();
			std::cout << "\nn = " << n << ", Cannon's runtime = " << splitTime << " ms" << std::endl;

			if (C == Cpar) {
				std::cout << "Parallel multiplication: correct result" << std::endl;
			} else {
				std::cout << "Parallel multiplication: invalid result" << std::endl;
				for (size_t i = 0; i < C.size(); i++) {
					if (C[i] != Cpar[i]) {
						std::cout << "i = " << i << ", C[i] = " << C[i] << ", Cpar[i] = " << Cpar[i] << std::endl;
					}
				}
			}
			if (C == Ccannon) {
				std::cout << "Cannon's algorithm     : correct result" << std::endl;
			} else {
				std::cout << "Cannon's algorithm     : invalid result" << std::endl;
				for (size_t i = 0; i < C.size(); i++) {
					if (C[i] != Ccannon[i]) {
						std::cout << "i = " << i << ", C[i] = " << C[i] << ", Ccannon[i] = " << Ccannon[i] << std::endl;
					}
				}
			}

		} else {
			MPI_Scatter(nullptr, nLocal2, MPI_INT, aLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatter(nullptr, nLocal2, MPI_INT, bLocal.data(), nLocal2, MPI_INT, 0, MPI_COMM_WORLD);

			// run MPI matrix multiplication
			if (blocking) cannonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);
			else cannonNonBlocking(aLocal.data(), bLocal.data(), cLocal.data(), nLocal, pSqrt);

			MPI_Gather(cLocal.data(), nLocal2, MPI_INT, nullptr, nLocal2, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}

	if (id == 0) {
		const double ts = swCPU.GetElapsedTimeMilliseconds();
		std::cout << "Sequential runtime = " << ts << " ms" << std::endl;
		{
			const double tp = swPAR.GetElapsedTimeMilliseconds();
			const double speedup = ts/tp;
			std::cout << "Parallel runtime   = " << tp << " ms, S = " << speedup << ", E = " << speedup/p << std::endl;
		}
		{
			const double tp = swMPI.GetElapsedTimeMilliseconds();
			const double speedup = ts/tp;
			std::cout << "Cannon's runtime   = " << tp << " ms, S = " << speedup << ", E = " << speedup/p << std::endl;
		}
	}

}