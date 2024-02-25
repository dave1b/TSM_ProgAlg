#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>
#include <thread>
#include <future>
#include <random>
#include "Stopwatch.h"

class Point {
	float x, y, z;

public:
	Point(float a, float b, float c) : x(a), y(b), z(c) {}

	bool operator==(const Point& p) const {
		return x == p.x && y == p.y && z == p.z;
	}

	bool operator<(const Point& p) const {
		return std::lexicographical_compare(&x, &x + 3, &p.x, &p.x + 3);
	}

	bool operator<=(const Point& p) const {
		return x <= p.x && y <= p.y && z <= p.z;
	}

	Point operator+(const Point& p) const {
		return { x + p.x, y + p.y, z + p.z };
	}

	friend std::ostream& operator<<(std::ostream& os, const Point& p) {
		return os << '(' << p.x << ',' << p.y << ',' << p.z << ')';
	}
};

//////////////////////////////////////////////////////////////////////////////////////////////
// Sequential range query
static std::vector<Point> rqSerial(std::vector<Point>& v, const Point& from, const Point& to) {
	std::vector<Point> result;

	std::for_each(v.begin(), v.end(), [from, to, &result](const Point& p) {
		if (from <= p && p <= to) result.push_back(p);
	});
	return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel range query
static std::vector<Point> rqPar1(std::vector<Point>& v, const Point& from, const Point& to) {
	std::vector<Point> result;

	// TODO use std::mutex for thread safe access of result
	return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Parallel range query
static std::vector<Point> rqPar2(std::vector<Point>& v, const Point& from, const Point& to) {
	const auto nThreads = std::thread::hardware_concurrency();

	std::vector<Point> result;

	// TODO: don't use synchronization, but serial reduction
	return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template<typename T>
static void check(const char text[], const T& ref, const T& result, double ts, double tp) {
	static const unsigned p = std::thread::hardware_concurrency();
	const double S = ts/tp;
	const double E = S/p;

	std::cout << std::setw(30) << std::left << text << result.size();
	std::cout << " in " << std::right << std::setw(6) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << ", E = " << E << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << (ref == result) << std::endl << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Different range query tests
void rangeQuery() {
	std::cout << "\nRange Query Tests" << std::endl;

	constexpr int N = 10'000'000;
	
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;
	Stopwatch sw;
	std::vector<Point> points;
	Point from(dist(e), dist(e), dist(e));
	Point to = from + Point(dist(e), dist(e), dist(e));
	//	Point to = from + Point(0.1f, 0.1f, 0.1f);

	points.reserve(N);
	for (int i = 0; i < N; i++) points.emplace_back(dist(e), dist(e), dist(e));

	sw.Start();
	std::vector<Point> resultS = rqSerial(points, from, to);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	std::sort(resultS.begin(), resultS.end());
	check("Sequential:", resultS, resultS, ts, ts);

	sw.Restart();
	std::vector<Point> result1 = rqPar1(points, from, to);
	sw.Stop();
	std::sort(result1.begin(), result1.end());
	check("Parallel query:", resultS, result1, ts, sw.GetElapsedTimeMilliseconds());

	sw.Restart();
	std::vector<Point> result2 = rqPar2(points, from, to);
	sw.Stop();
	std::sort(result2.begin(), result2.end());
	check("Parallel reduction:", resultS, result2, ts, sw.GetElapsedTimeMilliseconds());
}

