#pragma once

#include "Process.h"
#include "TaskGraph.h"
#include "Node.h"
#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>
#include <numeric>
#include <atomic>
#include <mutex>
#include <thread>

/// <summary>
/// Parallel searcher in DFS
/// </summary>
class DFSearcher {
	const std::vector<Task*>& m_sorted;		// topologically sorted tasks
	std::vector<DFSearcher>* m_searchers;	// array of all parallel searchers
	std::vector<Process> m_schedule;		// best schedule
	std::list<Node> m_openList;				// open list
	size_t m_size;							// number of tasks in graph
	size_t m_criticalPathLen;				// critical path length measured in task duration	
	volatile std::atomic_bool m_searching;	// true while searching
	std::atomic_bool m_outOfWork;			// true if needs new work
	std::atomic<int> m_bestSolution;		// current best solution
	mutable std::mutex m_mutex;				// for synchronized access to open list
	int m_myID;								// index of this searcher in m_searcher

public:
	DFSearcher(const TaskGraph& g)
		: m_sorted(g.m_sorted)
		, m_searchers(nullptr)
		, m_size(g.m_size)
		, m_criticalPathLen(g.m_criticalPathLen)
		, m_searching(true) // m_searching must be true before any thread starts running, otherwise a very fast thread could send stop before another thread starts searching
		, m_outOfWork(false)
		, m_bestSolution(-1)
		, m_myID(-1)
	{}
	DFSearcher(const DFSearcher& s)
		: m_sorted(s.m_sorted)
		, m_searchers(s.m_searchers)
		, m_size(s.m_size)
		, m_criticalPathLen(s.m_criticalPathLen)
		, m_bestSolution(-1)
		, m_myID(s.m_myID)
	{
		m_outOfWork.store(s.m_outOfWork);
		m_searching.store(s.m_searching); 
	}

	// getter
	std::vector<Process> getSchedule() const { return m_schedule; }

	// setter
	void setID(int id) { m_myID = id; }
	void setSearchers(std::vector<DFSearcher>* searchers) { m_searchers = searchers; }

	/// <summary>
	/// Add first node to open list
	/// </summary>
	/// <param name="p">number of processes</param>
	/// <param name="idx">index of the first task</param>
	void addInitalWork(size_t p, int idx) {
		m_openList.emplace_back(p, idx); 
		// save all task start times in added node
		for (auto t : m_sorted) m_openList.back().m_startTimes.push_back(t->m_start);
	}

	/// <summary>
	/// Start parallel searching
	/// </summary>
	/// <param name="bestSolution">current best solution (task schedule duration)</param>
	/// <returns>best duration</returns>
	int startSearching(int bestSolution);

	/// <summary>
	/// Stop searching
	/// </summary>
	void stopSearching() { m_searching.store(false); }

private:
	// threadsafe methods

	/// <summary>
	/// Return number of open nodes
	/// </summary>
	size_t openNodes() const {
		std::lock_guard<std::mutex> m_monitor(m_mutex);

		return m_openList.size();
	}

	/// <summary>
	/// Add node v to the open list
	/// </summary>
	void add(const Node& v) {
		std::lock_guard<std::mutex> m_monitor(m_mutex);

		m_openList.push_back(v);
	}

	/// <summary>
	/// Remove last node of the open list (if not empty)
	/// </summary>
	Node remove() {
		std::lock_guard<std::mutex> m_monitor(m_mutex);
		Node v;

		if (!m_openList.empty()) {
			v = std::move(m_openList.back());
			m_openList.pop_back();
		}
		return v;
	}

	/// <summary>
	/// Steal nodes from searcher s
	/// </summary>
	void stealWorkFrom(DFSearcher& s) {
		//std::scoped_lock monitor(m_mutex, s.m_mutex); // lock both open lists (needs C++17)

		auto lambda = [this, &s] {
			// TODO
		};

		// dead-lock prevention by ordered ressource allocation
		if (m_myID < s.m_myID) {
			std::lock_guard<std::mutex> monitor1(m_mutex);
			std::lock_guard<std::mutex> monitor2(s.m_mutex);

			lambda();
		} else {
			std::lock_guard<std::mutex> monitor2(s.m_mutex);
			std::lock_guard<std::mutex> monitor1(m_mutex);

			lambda();
		}
	}

	/// <summary>
	/// Broadcast duration of my best solution to all searchers but myself
	/// </summary>
	void broadcastBestSolution() const {
		for (auto& s : *m_searchers) {
			if (s.m_myID != m_myID && s.m_bestSolution > m_bestSolution) {
				s.m_bestSolution.store(m_bestSolution);
			}
		}
	}

	/// <summary>
	/// Broadcast stop to all searchers (including myself)
	/// </summary>
	void broadcastStop() {
		for (auto& s : *m_searchers) {
			s.stopSearching();
		}
	}

	/// <summary>
	/// Check if all searcher ran out of work
	/// </summary>
	void checkForEnd() {
		for (auto& s : *m_searchers) {
			if (!s.m_outOfWork) return;
		}
		stopSearching();
	}

	/// <summary>
	/// Sleep a while if the open list is empty
	/// </summary>
	void sleep() const {
		if (openNodes() == 0) {
			// still empty => wait
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
};