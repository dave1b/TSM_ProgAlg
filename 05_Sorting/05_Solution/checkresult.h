#pragma once

#include <iostream>
#include <iomanip>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template<typename T>
static void check(const char text[], T ref[], T result[], double ts, double tp, int n, unsigned p) {
	const double S = ts/tp;
	const double E = S/p;

	std::cout << std::setw(30) << std::left << text;
	std::cout << " in " << std::right << std::setw(8) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << ", E = " << E << std::endl;

	if (std::is_sorted(result, result + n)) {
		int i = 0;
		while (i < n && ref[i] == result[i]) i++;
		if (i < n) {
			std::cout << "array contains wrong elements" << std::endl;
		} else {
			std::cout << "array is correctly sorted" << std::endl;
		}
	} else {
		int i = 0;
		while (i < n && ref[i] == result[i]) i++;
		std::cout << "array is not sorted: at position " << i << " is " << result[i] << " instead of " << ref[i] << std::endl;
		for (int j = 0; j <= i; j++) std::cout << ref[j] << ' ';
		std::cout << std::endl;
		for (int j = 0; j <= i; j++) std::cout << result[j] << ' ';
		std::cout << std::endl;
		for (int j = i + 1; j < n; j++) std::cout << result[j] << ' ';
		std::cout << std::endl;
	}
}

