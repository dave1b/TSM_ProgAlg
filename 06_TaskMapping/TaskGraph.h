#pragma once

#include "Task.h"
#include "Process.h"
#include <vector>

//#define SHOW_PROGRESS

/// <summary>
/// DAG
/// </summary>
class TaskGraph {
	friend class BFSearcher;
	friend class DFSearcher;

	std::vector<Process> m_schedule;	// best schedule
	std::vector<Process> m_processes;	// current schedule
	std::vector<Task*> m_sorted;		// topologically sorted tasks
	size_t m_size;						// number of tasks
	size_t m_criticalPathLen;			// critical path length measured in task duration

public:
	TaskGraph(size_t p, size_t size) : m_processes(p), m_size(size), m_criticalPathLen(0) {}

	/// <summary>
	/// Find the best task mapping (schedule)
	/// </summary>
	/// <param name="root">root node of the task graph</param>
	/// <param name="searchAlgorithm">a search algorithm</param>
	/// <returns>duration of task schedule</returns>
	int findMapping(Task* root, int searchAlgorithm);

private:
	/// <summary>
	/// Topological sorting of tasks
	/// Store sorted tasks in m_sorted and set m_start of each task
	/// </summary>
	/// <param name="root">root node of the task graph</param>
	void topSort(Task* root);

	/// <summary>
	/// Set and return length of the critical path (longest path in task graph)
	/// </summary>
	int criticalPathLength() const;

	/// <summary>
	/// Return true if the schedule is feasible
	/// </summary>
	bool checkSchedule() const;

	/// <summary>
	/// Return number of communication steps (send operations) in m_schedule
	/// </summary>
	int communications() const;

	// search algorithms
	int searchSimple(int idx, int bestSolution);
	int searchBranchAndBound(int idx, int bestSolution);
	int searchOptimized(int idx, int bestSolution);
	int searchParallelDFS(int idx, int bestSolution);
	int searchParallelBFS(int idx, int bestSolution);
};