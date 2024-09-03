#pragma once

#include <iostream>
#include <iomanip>
#include <thread>

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template<typename T>
static void check(const char text[], const T& ref, const T& result, double ts, double tp) {
	static const unsigned p = std::thread::hardware_concurrency();
	const double S = ts/tp;
	const double E = S/p;

	std::cout << std::setw(30) << std::left << text << result;
	std::cout << " in " << std::right << std::setw(6) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << ", E = " << E << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl << std::endl;
}


