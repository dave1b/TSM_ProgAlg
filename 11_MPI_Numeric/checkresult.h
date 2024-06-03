#pragma once

#include <iostream>
#include <iomanip>
#include <thread>

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template<typename T>
static void check(const char text[], const T& ref, const T& result, double ts, double tp, bool verbose) {
	const double S = ts/tp;

	if (verbose) {
		std::cout << std::setw(40) << std::left << text << result.size();
		std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << std::endl;
		std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl;
	} else {
		std::cout << tp << ", " << std::boolalpha << (ref == result) << std::endl;
	}
}

