#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <vector>
#include <omp.h>
#include <random>
#include "Stopwatch.h"
#include "checkresult.h"

using Vector = std::vector<int>;

//////////////////////////////////////////////////////////////////////////////////////////////
// Standard implementation: Ts = O(n^3)
// Is very slow!
void matMultSeqStandard(const int a[], const int b[], int c[], const int n) {
	int ij = 0;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int bpos = j;
			int sum = 0;

			for (int k = 0; k < n; k++) {
				sum += a[k]*b[bpos];
				bpos += n;
			}
			c[ij++] = sum;
		}
		a += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Cache aware serial implementation
// Matrix C has to be initialized with 0 in advance
void matMultSeq(const int a[], const int b[], int c[], const int n) {
	// TODO
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
// Cache aware
void matMultPar(const int a[], const int b[], int c[], const int n) {
	// TODO use OMP
	#pragma omp parallel for default(none) shared(a, b, c, n)
	for (int i = 0; i < n; i++) {
		const int* arow = a + i*n;
		int* crow = c + i*n;
		int bpos = 0;

		for (int j = 0; j < n; j++) crow[j] = 0;
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				crow[j] += arow[k]*b[bpos++];
			}
		}
		crow += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
static void reset(Vector& v) {
	v.assign(v.size(), 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication tests
void matrixMultiplicationTests() {
	constexpr bool verbose = true;
	Stopwatch swCPU, swGPU;
	std::default_random_engine e;

	std::cout << std::endl << "Matrix multiplication Tests" << std::endl;
	
	for (int n = 1000; n <= 2000; n += 200) {
		if (verbose) {
			std::cout << std::endl << "matrix size: " << n << " x " << n << std::endl;
		} else {
			std::cout << n << std::endl;
		}

		const int n2 = n*n;
	    std::uniform_int_distribution<> dist(1, (int)sqrt(INT_MAX/n));
		Vector A(n2);
		Vector B(n2);
		Vector C(n2);
		Vector C2(n2);

		for (int i = 0; i < n2; i++) {
			A[i] = dist(e);
			B[i] = dist(e);
		}

        // run serial implementation: compute C
        swCPU.Start();
        matMultSeqStandard(A.data(), B.data(), C.data(), n);
        swCPU.Stop();
		const double ts = swCPU.GetElapsedTimeMilliseconds();
		std::cout << "Serial on CPU in " << ts << " ms" << std::endl;

        // run optimized serial implementation: compute C2
        swCPU.Restart();
        matMultSeq(A.data(), B.data(), C2.data(), n);
        swCPU.Stop();
		check("Serial cache aware:", C, C2, ts, swCPU.GetElapsedTimeMilliseconds(), verbose);
		reset(C2);

        // run parallel implementation: compute C2
        swCPU.Restart();
        matMultPar(A.data(), B.data(), C2.data(), n);
        swCPU.Stop();
		check("OMP:", C, C2, ts, swCPU.GetElapsedTimeMilliseconds(), verbose);
		reset(C2);
 	}
}
