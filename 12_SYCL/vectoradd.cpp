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

using Vector = std::vector<float>;

constexpr int BlockSize = 20;

//////////////////////////////////////////////////////////////////////////////////////////////
// serial vector addition
static void vectorAddition(const Vector &a, const Vector &b, Vector &c)
{
	for (size_t i = 0; i < a.size(); ++i)
	{
		c[i] = a[i] + b[i];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// parallel vector addition with OMP
static void vectorAdditionOMP(const Vector &a, const Vector &b, Vector &c)
{
#pragma omp parallel
#pragma omp master
	{
		std::cout << "number of processors = " << omp_get_num_procs() << std::endl;
		std::cout << "number of threads    = " << omp_get_num_threads() << std::endl;
	}

#pragma omp parallel for default(none) shared(a, b, c, std::cout)
	for (size_t i = 0; i < a.size(); ++i)
	{
		c[i] = a[i] + b[i];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// parallel vector addition with transform
static void vectorAdditionParallel(const Vector &a, const Vector &b, Vector &c)
{
	std::transform(std::execution::par, a.begin(), a.end(), b.begin(), c.begin(), [](auto ai, auto bi)
				   { return ai + bi; });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// GPU vector addition
static void vectorAdditionSYCL(sycl::queue &q, const Vector &a, const Vector &b, Vector &c)
{
	const sycl::nd_range<1> ndr(c.size(), BlockSize); // workgroup size = BlockSize

	sycl::buffer aBuf(a);
	sycl::buffer bBuf(b);
	sycl::buffer cBuf(c);

	q.submit([&](sycl::handler &h)
			 {
		sycl::accessor aAcc(aBuf, h, sycl::read_only);
		sycl::accessor bAcc(bBuf, h, sycl::read_only);
		sycl::accessor cAcc(cBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) { 
			const sycl::id<1> i = ii.get_global_id();

			cAcc[i] = aAcc[i] + bAcc[i]; 
		}); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// GPU vector addition (vectorized)
static void vectorAdditionSYCLvec(sycl::queue &q, const Vector &a, const Vector &b, Vector &c)
{
	using VectorType = sycl::float4;
	constexpr int VectorSize = VectorType::size();

	const sycl::range<1> r(a.size() / VectorSize);
	const sycl::nd_range<1> ndr(r, BlockSize); // workgroup size = BlockSize

	sycl::buffer<VectorType> aBuf((VectorType *)a.data(), r);
	sycl::buffer<VectorType> bBuf((VectorType *)b.data(), r);
	sycl::buffer<VectorType> cBuf((VectorType *)c.data(), r);

	q.submit([&](sycl::handler &h)
			 {
		sycl::accessor aAcc(aBuf, h, sycl::read_only);
		sycl::accessor bAcc(bBuf, h, sycl::read_only);
		sycl::accessor cAcc(cBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) { 
			const sycl::id<1> i = ii.get_global_id();

			cAcc[i] = aAcc[i] + bAcc[i]; 
		}); });
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
static void check(const char text[], const Vector &ref, const Vector &result, double ts, double tp)
{
	const double S = ts / tp;

	std::cout << std::setw(40) << std::left << text << result.size();
	std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
static void reset(Vector &v)
{
	v.assign(v.size(), 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void vectorAdditionTests()
{
	std::cout << "Vector Addition Tests" << std::endl;

	constexpr int N = 100'000'000;

	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;

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

	Vector a(N);
	Vector b(N);
	Vector r1(N);
	Vector r2(N);
	Stopwatch sw;

	for (int i = 0; i < N; ++i)
	{
		a[i] = dist(e);
		b[i] = dist(e);
	};

	sw.Start();
	vectorAddition(a, b, r1);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	std::cout << std::endl
			  << "Serial on CPU in " << ts << " ms" << std::endl;

	sw.Restart();
	vectorAdditionParallel(a, b, r2);
	sw.Stop();
	std::cout << std::endl;
	check("Parallel on CPU: ", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	reset(r2);

	sw.Restart();
	vectorAdditionOMP(a, b, r2);
	sw.Stop();
	std::cout << std::endl;
	check("OMP on CPU: ", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	reset(r2);

	// GPU processing: run-time is mainly memory-transfer time
	auto selector = sycl::default_selector_v; // The default device selector will select the most performant device.
	// auto selector = sycl::aspect_selector(sycl::aspect::cpu); // uses the CPU as the underlying OpenCL device
	sycl::queue q(selector, exception_handler);

	std::cout << std::endl
			  << "SYCL on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
	std::cout << "native vector width for float is   : " << q.get_device().get_info<sycl::info::device::native_vector_width_float>() << std::endl;
	std::cout << "preferred vector width for float is: " << q.get_device().get_info<sycl::info::device::preferred_vector_width_float>() << std::endl;

	try
	{
		sw.Restart();
		vectorAdditionSYCL(q, a, b, r2);
		q.wait(); // wait until compute tasks on GPU done
		sw.Stop();
		std::cout << std::endl;
		check("GPU:", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	}
	catch (const std::exception &e)
	{
		std::cout << "An exception is caught for vector add: " << e.what() << std::endl;
	}
	reset(r2);

	try
	{
		sw.Restart();
		vectorAdditionSYCLvec(q, a, b, r2);
		q.wait(); // wait until compute tasks on GPU done
		sw.Stop();
		std::cout << std::endl;
		check("GPU vectorized:", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	}
	catch (const std::exception &e)
	{
		std::cout << "An exception is caught for vector add: " << e.what() << std::endl;
	}
	reset(r2);
}
