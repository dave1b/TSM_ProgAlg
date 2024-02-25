#pragma once

#include <chrono>

/*
 Stopwatch measuring wall-clock time.
 CPU time could be measured with std::clock_t startcputime = std::clock();
 */
class Stopwatch {
	using Clock = std::chrono::high_resolution_clock;
	//using Clock = std::chrono::system_clock;

	Clock::time_point m_start;
	Clock::duration m_elapsed;
	bool m_isRunning;

public:
	Stopwatch()
		: m_elapsed{ 0 }
		, m_isRunning{ false } 
	{}

	void Start() {
		if (!m_isRunning) {
			m_start = Clock::now();
			m_isRunning = true;
		}
	}
	void Stop() {
		if (m_isRunning) {
			m_elapsed += Clock::now() - m_start;
			m_isRunning = false;
		}
	}
	void Reset() {
		m_elapsed = Clock::duration::zero();
		m_isRunning = false;
	}
	void Restart() {
		Reset();
		Start();
	}
	Clock::duration GetSplitTime() const {
		Clock::duration result(0);
		if (m_isRunning) {
			result = Clock::now() - m_start;
		}
		return result;
	}
	double GetSplitTimeSeconds() const {
		using sec = std::chrono::duration<double>;
		return std::chrono::duration_cast<sec>(GetSplitTime()).count();
	}
	double GetSplitTimeMilliseconds() const {
		using ms = std::chrono::duration<double, std::milli>;
		return std::chrono::duration_cast<ms>(GetSplitTime()).count();
	}
	long long GetSplitTimeNanoseconds() const {
		return std::chrono::nanoseconds(GetSplitTime()).count();
	}
	Clock::duration GetElapsedTime() const {
		Clock::duration result = m_elapsed;
		if (m_isRunning) {
			result += Clock::now() - m_start;
		}
		return result;
	}
	double GetElapsedTimeSeconds() const {
		using sec = std::chrono::duration<double>;
		return std::chrono::duration_cast<sec>(GetElapsedTime()).count();
	}
	double GetElapsedTimeMilliseconds() const {
		using ms = std::chrono::duration<double, std::milli>;
		return std::chrono::duration_cast<ms>(GetElapsedTime()).count();
	}
	long long GetElapsedTimeNanoseconds() const {
		return std::chrono::nanoseconds(GetElapsedTime()).count();
	}
};
