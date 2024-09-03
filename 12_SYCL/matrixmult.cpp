#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <vector>
#include <omp.h>
#include <random>
#include <sycl/sycl.hpp>
#include "Stopwatch.h"

using Vector = std::vector<int>;

constexpr int BlockSize = 20;

//////////////////////////////////////////////////////////////////////////////////////////////
// Cache aware serial implementation: Ts = O(n^3)
// Matrix C has to be initialized with 0 in advance
static void matMultSeq(const int a[], const int b[], int c[], const int n)
{
	int *crow = c;

	for (int i = 0; i < n; i++)
	{
		int bpos = 0;

		for (int k = 0; k < n; k++)
		{
			for (int j = 0; j < n; j++)
			{
				crow[j] += a[k] * b[bpos++];
			}
		}
		a += n;
		crow += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel matrix multiplication
// Cache aware parallel implementation: Ts = O(n^3)
// Matrix C has to be initialized with 0 in advance
static void matMultPar(const int a[], const int b[], int c[], const int n)
{
#pragma omp parallel for default(none) shared(a, b, c, n)
	for (int i = 0; i < n; i++)
	{
		const int *const arow = a + i * n;
		int *crow = c + i * n;
		int bpos = 0;

		for (int j = 0; j < n; j++)
			crow[j] = 0;
		for (int k = 0; k < n; k++)
		{
			for (int j = 0; j < n; j++)
			{
				crow[j] += arow[k] * b[bpos++];
			}
		}
		crow += n;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel matrix multiplication on GPU
// Use nd_range, because it is necessary to get good GPU performance.
// classical approach with p = n^2 work-items
// output data partitioning
// each work-item computes a dot-product of two vectors of size n
// Tp = O(n), Cost = O(n^3)
static void matMultSYCL(sycl::queue &q, const Vector &a, const Vector &b, Vector &c, const int n)
{
	const sycl::range<2> r(n, n);
	const sycl::nd_range<2> ndr(r, {BlockSize, BlockSize}); // workgroup size: 20x20

	sycl::buffer<int, 2> aBuf(a.data(), r);
	sycl::buffer<int, 2> bBuf(b.data(), r);
	sycl::buffer<int, 2> cBuf(c.data(), r);

	// submit matrix multiplication
	q.submit([&](sycl::handler &h)
			 {
		sycl::accessor aAcc(aBuf, h, sycl::read_only);
		sycl::accessor bAcc(bBuf, h, sycl::read_only);
		sycl::accessor cAcc(cBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) { 
			const sycl::id<2> i = ii.get_global_id();
            const int row = i[0];
            const int col = i[1];
			//std::printf("%d %d\n", row, col);

            int sum = 0;

            for(int k = 0; k < n; k++) {
			    sum += aAcc[row][k]*bAcc[k][col]; 
            }

            cAcc[i] = sum;
		}); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel matrix multiplication on GPU with vectorizawion
// Use nd_range, because it is necessary to get good GPU performance.
static void matMultSYCLvec(sycl::queue &q, const Vector &a, const Vector &b, Vector &c, const int n)
{
	using VectorType = sycl::int8;
	constexpr int VectorSize = VectorType::size();

	const int nd8 = n / VectorSize;
	const sycl::range<2> r(n, n);
	const sycl::range<2> r2(n, nd8);
	const sycl::nd_range<2> ndr(r, {BlockSize, BlockSize}); // workgroup size: 20x20

	sycl::buffer<VectorType, 2> aBuf((VectorType *)a.data(), r2);
	sycl::buffer<int, 2> bBuf(b.data(), r);
	sycl::buffer<int, 2> cBuf(c.data(), r);

	// TODO use SYCL
	// submit matrix multiplication
	q.submit([&](sycl::handler &h)
			 {
		sycl::accessor aAcc(aBuf, h, sycl::read_only);
		sycl::accessor bAcc(bBuf, h, sycl::read_only);
		sycl::accessor cAcc(cBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](sycl::nd_item<2> ii) { 
			const sycl::id<2> i = ii.get_global_id();
            const int row = i[0];
            const int col = i[1];
			VectorType s{};

            for(int k = 0; k < nd8; k++) {
				const int bRow = k*VectorSize;

				VectorType bc(
					bAcc[bRow + 0][col],
					bAcc[bRow + 1][col],
					bAcc[bRow + 2][col],
					bAcc[bRow + 3][col],
					bAcc[bRow + 4][col],
					bAcc[bRow + 5][col],
					bAcc[bRow + 6][col],
					bAcc[bRow + 7][col]
				);
			    s += aAcc[row][k]*bc; 
            }

            cAcc[i] = s.s0() + s.s1() + s.s2() + s.s3() + s.s4() + s.s5() + s.s6() + s.s7();
		}); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel matrix multiplication on GPU
// Classical approach with p = n^2 work-items and using local memory
// block partitioning of A and B: block size = work-group size
// Aloc and Bloc is local memory to store one block each (similar to Cannon's algorithm)
// output data partitioning
// each work-item computes a dot-product of two vectors of size n
// Tp = O(n), Cost = O(n^3)
static void matMultSYCLloc(sycl::queue &q, const Vector &a, const Vector &b, Vector &c, const int n)
{
	const sycl::range<2> r(n, n);
	const sycl::range<2> wg(BlockSize, BlockSize); // workgroup size: 20x20
	const sycl::nd_range<2> ndr(r, wg);

	sycl::buffer<int, 2> aBuf(a.data(), r);
	sycl::buffer<int, 2> bBuf(b.data(), r);
	sycl::buffer<int, 2> cBuf(c.data(), r);

	// TODO use SYCL
	// submit matrix mulwiplication
	q.submit([&](sycl::handler &h)
			 {
		sycl::accessor aAcc(aBuf, h, sycl::read_only);
		sycl::accessor bAcc(bBuf, h, sycl::read_only);
		sycl::accessor cAcc(cBuf, h, sycl::write_only, sycl::no_init);
		sycl::local_accessor<int, 2> aLocAcc(wg, h); // local memory accessor
		sycl::local_accessor<int, 2> bLocAcc(wg, h); // local memory accessor

		h.parallel_for(ndr, [=](sycl::nd_item<2> ii) { 
			// grid index
			const sycl::id<2> gi = ii.get_global_id();

			// group range
			const sycl::range<2> gr = ii.get_group().get_group_range();

			// work item index in group
			const sycl::id<2> wi = ii.get_local_id();

			// Cxy is used to store the element of the block sub-matrix
			// that is computed by the thread
			int Cxy = 0;

			// Loop over all the sub-matrices of A and B
			// required to compute the block sub-matrix
			for (int groupIdx = 0; groupIdx < gr[0]; ++groupIdx) {
				// Load the matrices from global memory to local memory; 
				// each thread loads one element of each matrix
				aLocAcc[wi] = aAcc[gi[0]][groupIdx*BlockSize + wi[1]];
				bLocAcc[wi] = bAcc[groupIdx*BlockSize + wi[0]][gi[1]];

				// Synchronize to make sure the local sub-matrices are loaded
				ii.barrier(sycl::access::fence_space::local_space);

				// Multiply the two matrices together;
				// each thread computes one element of the block sub-matrix
#pragma unroll
				for (int k = 0; k < BlockSize; ++k)
					Cxy += aLocAcc[wi[0]][k]*bLocAcc[k][wi[1]];

				// Synchronize to make sure that the preceding
				// computation is done before loading two new
				// sub-matrices of A and B in the next iteration
				ii.barrier(sycl::access::fence_space::local_space);
			}

            cAcc[gi] = Cxy;
		}); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template <typename T>
static void check(const char text[], const T &ref, const T &result, double ts, double tp, bool verbose)
{
	const double S = ts / tp;

	if (verbose)
	{
		std::cout << std::setw(40) << std::left << text << result.size();
		std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << std::endl;
		std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl;
	}
	else
	{
		std::cout << tp << ", " << std::boolalpha << (ref == result) << std::endl;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
static void reset(Vector &v)
{
	v.assign(v.size(), 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication tests
void matrixMultiplicationTests()
{
	constexpr bool verbose = true;
	Stopwatch swCPU, swGPU;
	std::default_random_engine e;

	// Create an exception handler for asynchronous SYCL exceptions
	auto exception_handler = [](sycl::exception_list e_list)
	{
		for (std::exception_ptr const &e : e_list)
		{
			try
			{
				std::rethrow_exception(e);
			}
			catch (std::exception const &e)
			{
#if _DEBUG
				std::cout << "Failure" << std::endl;
#endif
				std::terminate();
			}
		}
	};

	std::cout << std::endl
			  << "Matrix multiplication Tests" << std::endl;

	auto selector = sycl::default_selector_v; // The default device selector will select the most performant device.
	// auto selector = sycl::aspect_selector(sycl::aspect::cpu); // uses the CPU as the underlying OpenCL device
	sycl::queue q(selector, exception_handler);
	// sycl::queue q;

	if (verbose)
	{
		std::cout << "SYCL on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
	}

	for (int n = 1000; n <= 2000; n += 200)
	{
		if (verbose)
		{
			std::cout << std::endl
					  << "matrix size: " << n << " x " << n << std::endl;
		}
		else
		{
			std::cout << n << std::endl;
		}

		const int n2 = n * n;
		std::uniform_int_distribution<> dist(1, (int)sqrt(INT_MAX / n));
		Vector A(n2);
		Vector B(n2);
		Vector C(n2);
		Vector Cpar(n2);

		for (int i = 0; i < n2; i++)
		{
			A[i] = dist(e);
			B[i] = dist(e);
		}

		// run serial implementation: compute C
		swCPU.Start();
		matMultSeq(A.data(), B.data(), C.data(), n);
		swCPU.Stop();
		const double ts = swCPU.GetElapsedTimeMilliseconds();
		std::cout << "Serial on CPU in " << ts << " ms" << std::endl;

		// run parallel implementation: compute Cpar
		swCPU.Restart();
		matMultPar(A.data(), B.data(), Cpar.data(), n);
		swCPU.Stop();
		check("OMP:", C, Cpar, ts, swCPU.GetElapsedTimeMilliseconds(), verbose);
		reset(Cpar);

		// run parallel implementations (SYCL)
		try
		{
			swGPU.Restart();
			matMultSYCL(q, A, B, Cpar, n);
			// matMultSYCL_crash(q, A, B, Cpar, n);
			q.wait();
			swGPU.Stop();
			check("GPU with p = n^2:", C, Cpar, ts, swGPU.GetElapsedTimeMilliseconds(), verbose);
			reset(Cpar);
		}
		catch (const std::exception &e)
		{
			std::cout << "An exception is caught for matrix multiplication: " << e.what() << std::endl;
		}

		try
		{
			swGPU.Restart();
			matMultSYCLvec(q, A, B, Cpar, n);
			q.wait();
			swGPU.Stop();
			check("GPU vectorized with p = n^2:", C, Cpar, ts, swGPU.GetElapsedTimeMilliseconds(), verbose);
			reset(Cpar);
		}
		catch (const std::exception &e)
		{
			std::cout << "An exception is caught for matrix multiplication: " << e.what() << std::endl;
		}

		try
		{
			swGPU.Restart();
			matMultSYCLloc(q, A, B, Cpar, n);
			q.wait();
			swGPU.Stop();
			check("GPU with local memory with p = n^2:", C, Cpar, ts, swGPU.GetElapsedTimeMilliseconds(), verbose);
			reset(Cpar);
		}
		catch (const std::exception &e)
		{
			std::cout << "An exception is caught for matrix multiplication: " << e.what() << std::endl;
		}
	}
}
