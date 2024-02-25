#pragma once

#include <list>
#include <initializer_list>
#include <queue>
#include <algorithm>

/// <summary>
/// Task in a task graph
/// </summary>
struct Task {
	std::list<Task*> m_tasks;	// dependent tasks
	int m_id;					// 1-based ID
	int m_start;				// start time
	int m_duration;				// duration
	int m_inDeg;				// task graph in-degree
	int m_idx;					// index in topologically sorted array (only used in parallel search)

	Task(int id = 0, int duration = 1)
		: m_id(id)
		, m_start(-1)
		, m_duration(duration)
		, m_inDeg(0)
		, m_idx(-1)
	{}

	/// <summary>
	/// Task ID for output
	/// </summary>
	char id() const { return 'A' + m_id - 1; }

	/// <summary>
	/// Add one dependent task
	/// </summary>
	void addTask(Task* t) {
		t->m_inDeg++;
		m_tasks.push_back(t);
	}

	/// <summary>
	/// Add a set of dependent tasks
	/// </summary>
	void addTasks(std::initializer_list<Task*> tasks) {
		for (auto it = tasks.begin(); it < tasks.end(); it++) {
			addTask(*it);
		}
	}

	/// <summary>
	/// Return true if task t is dependent of this task
	/// </summary>
	bool isPredecessorOf(Task* t) const {
		return std::count(m_tasks.begin(), m_tasks.end(), t) > 0;
	}

	/// <summary>
	/// Update the task start time of this task and its dependent tasks
	/// </summary>
	/// <param name="start">task start time</param>
	void updateStart(int start) {
		m_start = std::max(m_start, start);

		const int succMinStart = m_start + m_duration;

		for (auto it = m_tasks.begin(); it != m_tasks.end(); it++) {
			(*it)->updateStart(succMinStart);
		}
	}
};