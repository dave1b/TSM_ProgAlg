#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include "Task.h"

/// <summary>
/// A scheduled process containing of a timely ordered list of tasks.
/// </summary>
class Process {
	std::vector<Task*> m_tasks;			// scheduled tasks in correct order
	int m_duration;						// process duration

public:
	static Task* s_currentTask;			// used in operator<

	Process() : m_duration(0) {}

	int getDuration() const { return m_duration; }
	void addDuration(int dur) { m_duration += dur; }
	bool hasPredecessorOf(Task* t) const { return (m_tasks.empty()) ? true : m_tasks.back()->isPredecessorOf(t); }
	size_t count(Task* t) const { return std::count(m_tasks.begin(), m_tasks.end(), t); }

	/// <summary>
	/// Returns the number of inter process communications (send operations)
	/// </summary>
	int communications() const {
		int sum = 0;

		// iterate all mapped tasks t
		for (auto tit = m_tasks.begin(); tit != m_tasks.end(); tit++) {
			Task* t = *tit;

			// check for each dependent task of t if it has been mapped to this process
			for (auto it = t->m_tasks.begin(); it != t->m_tasks.end(); it++) {
				if (count(*it) == 0) sum++;
			}
		}
		return sum;
	}

	/// <summary>
	/// Comparator used for ordering processes: smaller duration and predecessor task in the same process are relevant
	/// </summary>
	/// <param name="p">rhs</param>
	bool operator<(const Process& p) const {
		if (m_duration == p.m_duration) {
			return hasPredecessorOf(s_currentTask) && !p.hasPredecessorOf(s_currentTask);
		} else {
			return m_duration < p.m_duration;
		}
	}

	/// <summary>
	/// Append waiting time
	/// </summary>
	Process& operator+=(int dur) {
		assert(dur >= 0);
		assert(m_duration >= 0);
		m_duration += dur;
		return *this;
	}

	/// <summary>
	/// Remove waiting time
	/// </summary>
	Process& operator-=(int dur) {
		assert(dur >= 0);
		m_duration -= dur;
		assert(m_duration >= 0);
		return *this;
	}

	/// <summary>
	/// Append task
	/// </summary>
	Process& operator+=(Task* t) {
		assert(t);
		m_tasks.push_back(t);
		return *this;
	}

	/// <summary>
	/// Remove last task
	/// </summary>
	Process& operator-=(Task* t) {
		assert(t && !m_tasks.empty());
		m_tasks.pop_back();
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& os, const Process& p) {
		int time = 0;
		for (auto it = p.m_tasks.cbegin(); it != p.m_tasks.cend(); it++) {
			const Task* t = *it;

			for (; time < t->m_start; time++) {
				os << ' ';
			}
			for (int i = 0; i < t->m_duration; i++) {
				os << t->id();
				time++;
			}
		}
		return os;
	}

	/// <summary>
	/// Compute maximum duration for a set of processes
	/// </summary>
	friend int duration(const std::vector<Process>& processes) {
		int max = 0;

		// find maximum duration over all processes
		for (auto it = processes.begin(); it < processes.end(); it++) {
			const Process& p = *it;

			if (p.m_duration > max) max = p.m_duration;
		}
		return max;
	}

};