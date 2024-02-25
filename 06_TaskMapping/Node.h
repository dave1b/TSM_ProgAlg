#pragma once

#include "Process.h"

class BFSearcher;
class DFSearcher;

/// <summary>
/// Necessary information in parallel search
/// </summary>
class Node {
	friend class BFSearcher;
	friend class DFSearcher;

protected:
	std::vector<Process> m_processes;	// current schedule
	std::vector<int> m_startTimes;		// current task start times for all tasks
	int m_idx;							// next task

public:
	/// <summary>
	/// ctor
	/// </summary>
	/// <param name="p">number of processes</param>
	/// <param name="idx">index of the next task</param>
	Node(size_t p = 0, int idx = -1)
		: m_processes(p)
		, m_idx(idx)
	{}
	/// <summary>
	/// ctor
	/// </summary>
	/// <param name="processes">schedule</param>
	/// <param name="idx">index of the next task</param>
	Node(const std::vector<Process>& processes, int idx)
		: m_processes(processes)
		, m_idx(idx)
	{}
	Node(const Node& v)
		: m_processes(v.m_processes)
		, m_startTimes(v.m_startTimes)
		, m_idx(v.m_idx)
	{}
		Node(Node&& v) noexcept
		: m_processes(std::move(v.m_processes))
		, m_startTimes(std::move(v.m_startTimes))
		, m_idx(v.m_idx)
	{
		v.m_idx = -1;
	}
	Node& operator=(Node&& v) noexcept {
		if (&v != this) {
			m_processes = std::move(v.m_processes);
			m_startTimes = std::move(v.m_startTimes);
			m_idx = v.m_idx; v.m_idx = -1;
		}
		return *this;
	}

	/// <summary>
	/// Node contains valid information
	/// </summary>
	bool isValid() const { return m_processes.size() > 0; }

	/// <summary>
	/// Update the task start times in m_startTimes
	/// </summary>
	/// <param name="tasks">toplogically sorted list of tasks</param>
	/// <param name="idx">index of the current task in tasks</param>
	/// <param name="start">task start time of task with index idx</param>
	void updateStartTimes(const std::vector<Task*>& tasks, int idx, int start) {
		// update task starting time of task t
		m_startTimes[idx] = std::max(m_startTimes[idx], start);

		const Task* t = tasks[idx]; assert(t->m_idx == idx);
		const int succMinStart = m_startTimes[idx] + t->m_duration;

		// update task starting time of the children of t
		for (auto it = t->m_tasks.begin(); it != t->m_tasks.end(); it++) {
			updateStartTimes(tasks, (*it)->m_idx, succMinStart);
		}
	}
};

