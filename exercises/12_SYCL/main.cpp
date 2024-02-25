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

//////////////////////////////////////////////////////////////////////////////////////////////
// serial vector addition
static void vectorAddition(const Vector& a, const Vector& b, Vector& c) {
	for (size_t i = 0; i < a.size(); ++i) {
		c[i] = a[i] + b[i];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// parallel vector addition with OMP
static void vectorAdditionOMP(const Vector& a, const Vector& b, Vector& c) {
	// TODO use OMP
}

//////////////////////////////////////////////////////////////////////////////////////////////
// parallel vector addition with transform
static void vectorAdditionParallel(const Vector& a, const Vector& b, Vector& c) {
	// TODO use std::transform
}

//////////////////////////////////////////////////////////////////////////////////////////////
// GPU vector addition
static void vectorAdditionSYCL(sycl::queue& q, const Vector& a, const Vector& b, Vector& c) {
	// TODO use SYCL
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
static void check(const char text[], const Vector& ref, const Vector& result, double ts, double tp) {
	const double S = ts/tp;

	std::cout << std::setw(40) << std::left << text << result.size();
	std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	std::cout << "Vector Addition Tests" << std::endl;

	constexpr int N = 100'000'000;

	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;

	// Create an exception handler for asynchronous SYCL exceptions
	auto exception_handler = [](sycl::exception_list e_list) {
		for (std::exception_ptr const& e : e_list) {
			try {
				std::rethrow_exception(e);
			} catch (std::exception const& e) {
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
	Stopwatch sw;

	for (int i = 0; i < N; ++i) {
		a[i] = dist(e);
		b[i] = dist(e);
	};

	std::cout << std::endl;
	sw.Start();
	vectorAddition(a, b, r1);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	std::cout << "Serial on CPU in " << ts << " ms" << std::endl;

	{
		Vector r2(N);
		std::cout << std::endl;
		sw.Restart();
		vectorAdditionOMP(a, b, r2);
		sw.Stop();
		check("OMP on CPU: ", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	}
	{
		Vector r2(N);
		std::cout << std::endl;
		sw.Restart();
		vectorAdditionParallel(a, b, r2);
		sw.Stop();
		check("Parallel on CPU: ", r1, r2, ts, sw.GetElapsedTimeMilliseconds());
	}
	{
		Vector r2(N);
		std::ostringstream oss;

		try {
			auto selector = sycl::default_selector_v; // The default device selector will select the most performant device.
			//auto selector = sycl::aspect_selector(sycl::aspect::cpu); // uses the CPU as the underlying OpenCL device
			sycl::queue q(selector, exception_handler);
			std::cout << std::endl;
			sw.Restart();
			vectorAdditionSYCL(q, a, b, r2);
			q.wait(); // wait until compute tasks on GPU done
			sw.Stop();
			oss << "SYCL on " << q.get_device().get_info<sycl::info::device::name>() << ": ";
			check(oss.str().c_str(), r1, r2, ts, sw.GetElapsedTimeMilliseconds());
		} catch (const std::exception& e) {
			std::cout << "An exception is caught for vector add: " << e.what() << std::endl;
  		}
	}
}
