#pragma once

#include <chrono>

/// <summary>
/// Stopwatch
/// Typical usages 
/// - re-usage of the same instance:  Start - Stop - GetElapsedTime ... Restart - Stop - GetElapsedTime 
/// - cummulative duration:           Start - Stop ... Start - Stop - GetElapsedTime
/// - long duration with split-times: Start - GetSplitTime - GetSplitTime - Stop - GetElapsedTime
/// - long duration with intervals  : Start - GetIntervalTime - GetIntervalTime - Stop - GetElapsedTime
/// </summary>
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

	/// <summary>
	/// Start stopwatch. No effect if the stopwatch is already running.
	/// </summary>
	void Start() {
		if (!m_isRunning) {
			m_start = Clock::now();
			m_isRunning = true;
		}
	}

	/// <summary>
	/// Stop stopwatch. Updates elapsed time. No effect if the stopwatch isn't running.
	/// </summary>
	void Stop() {
		if (m_isRunning) {
			m_elapsed += Clock::now() - m_start;
			m_isRunning = false;
		}
	}

	/// <summary>
	/// Stops running stopwatch and resets elapsed time to zero.
	/// </summary>
	void Reset() {
		m_isRunning = false;
		m_elapsed = Clock::duration::zero();
	}

	/// <summary>
	/// Reset elapsed time and start stopwatch again.
	/// </summary>
	void Restart() {
		Reset();
		Start();
	}


	/// <summary>
	/// Return split time. No effect if the stopwatch isn't running.
	/// </summary>
	Clock::duration GetSplitTime() const {
		if (m_isRunning) {
			return Clock::now() - m_start;
		} else {
			return Clock::duration::zero();
		}
	}
	/// <summary>
	/// Return split time in seconds. No effect if the stopwatch isn't running.
	/// </summary>
	double GetSplitTimeSeconds() const {
		using sec = std::chrono::duration<double>;
		return std::chrono::duration_cast<sec>(GetSplitTime()).count();
	}
	/// <summary>
	/// Return split time in milliseconds. No effect if the stopwatch isn't running.
	/// </summary>
	double GetSplitTimeMilliseconds() const {
		using ms = std::chrono::duration<double, std::milli>;
		return std::chrono::duration_cast<ms>(GetSplitTime()).count();
	}
	/// <summary>
	/// Return split time in nanoseconds. No effect if the stopwatch isn't running.
	/// </summary>
	long long GetSplitTimeNanoseconds() const {
		return std::chrono::nanoseconds(GetSplitTime()).count();
	}


	/// <summary>
	/// Return interval time. No effect if the stopwatch isn't running.
	/// Combination of GetSplitTime - Stop - Start
	/// </summary>
	Clock::duration GetIntervalTime() {
		if (m_isRunning) {
			const Clock::time_point start = Clock::now();
			const Clock::duration interval = start - m_start;

			m_elapsed += interval;
			m_start = start;
			return interval;
		} else {
			return Clock::duration::zero();
		}
	}
	/// <summary>
	/// Return interval time in seconds. No effect if the stopwatch isn't running.
	/// Combination of GetSplitTime - Stop - Start
	/// </summary>
	double GetIntervalTimeSeconds() {
		using sec = std::chrono::duration<double>;
		return std::chrono::duration_cast<sec>(GetIntervalTime()).count();
	}
	/// <summary>
	/// Return interval time in milliseconds. No effect if the stopwatch isn't running.
	/// Combination of GetSplitTime - Stop - Start
	/// </summary>
	double GetIntervalTimeMilliseconds() {
		using ms = std::chrono::duration<double, std::milli>;
		return std::chrono::duration_cast<ms>(GetIntervalTime()).count();
	}
	/// <summary>
	/// Return interval time in nanoseconds. No effect if the stopwatch isn't running.
	/// Combination of GetSplitTime - Stop - Start
	/// </summary>
	long long GetIntervalTimeNanoseconds() {
		return std::chrono::nanoseconds(GetIntervalTime()).count();
	}


	/// <summary>
	/// Return elapsed time since first start after reset.
	/// </summary>
	Clock::duration GetElapsedTime() const {
		if (m_isRunning) {
			return m_elapsed + Clock::now() - m_start;
		} else {
			return m_elapsed;
		}
	}
	/// <summary>
	/// Stop stopwatch and return elapsed time in seconds.
	/// </summary>
	double GetElapsedTimeSeconds() const {
		using sec = std::chrono::duration<double>;
		return std::chrono::duration_cast<sec>(GetElapsedTime()).count();
	}
	/// <summary>
	/// Stop stopwatch and return elapsed time in milliseconds.
	/// </summary>
	double GetElapsedTimeMilliseconds() const {
		using ms = std::chrono::duration<double, std::milli>;
		return std::chrono::duration_cast<ms>(GetElapsedTime()).count();
	}
	/// <summary>
	/// Stop stopwatch and return elapsed time in nanoseconds.
	/// </summary>
	long long GetElapsedTimeNanoseconds() const {
		return std::chrono::nanoseconds(GetElapsedTime()).count();
	}
};
