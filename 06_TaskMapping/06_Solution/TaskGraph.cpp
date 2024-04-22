#include "TaskGraph.h"
#include "Process.h"
#include "DFSearcher.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <future>
#include <array>
#include <thread>

Task* Process::s_currentTask = nullptr;

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Topological sorting of tasks
/// Store sorted tasks in m_sorted and set m_start of each task
/// Approach: choose node with indeg = 0 and among all nodes with indeg = 0 a node with smallest start time
/// </summary>
/// <param name="root">root node of the task graph</param>
void TaskGraph::topSort(Task* root) {
	std::vector<Task*> open(m_size + 1, nullptr); // candidates to check for indeg = 0; +1, because task-id is 1-based
	m_sorted.reserve(m_size);

	Task* t = root;
	t->m_start = 0;

	// store root node (task) in m_sorted
	m_sorted.push_back(t);
	t->m_idx = (int)m_sorted.size() - 1; // store position of t in m_sorted

	while (m_sorted.size() < m_size) {
		assert(t);

		// iterate all dependent tasks of t
		for (auto it = t->m_tasks.begin(); it != t->m_tasks.end(); it++) {
			Task* t2 = *it; // successor of t

			open[t2->m_id] = t2;	// store candidate
			t2->m_inDeg--;			// decrease indeg, because task t (a predecessor of t2) has been added to m_sorted
			t2->m_start = std::max(t2->m_start, t->m_start + t->m_duration);	// update task start time
		}

		// find a task with indeg = 0 and with smallest start time
		int start = INT32_MAX;
		size_t best;
		t = nullptr;

		for (size_t i = 1; i <= m_size; i++) {
			if (open[i] && open[i]->m_inDeg == 0) {
				Task* t2 = open[i];

				if (t2->m_start < start) {
					start = t2->m_start;
					best = i;
					t = t2;
				}
			}
		}

		// add task t to m_sorted and delete it from the open list of candidates
		if (t) {
			m_sorted.push_back(t);
			t->m_idx = (int)m_sorted.size() - 1; // store position of t in m_sorted
			open[best] = nullptr;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Set and return length of the critical path (longest path in task graph)
/// Approach: process tasks in topologically sorted order and update path length of children
/// </summary>
int TaskGraph::criticalPathLength() const {
	std::vector<int> lengths(m_size + 1, 0); // length from root to the other nodes, +1, because task-id is 1-based

	for (auto it = m_sorted.begin(); it < m_sorted.end(); it++) {
		Task* current = *it;

		// update length of current task
		if (!lengths[current->m_id]) lengths[current->m_id] = current->m_duration;

		for (auto succ = current->m_tasks.begin(); succ != current->m_tasks.end(); succ++) {
			Task* successor = *succ;

			// length[s] = max(length[s], length[c] + dur(c,s))
			lengths[successor->m_id] = std::max(lengths[successor->m_id], lengths[current->m_id] + successor->m_duration);
		}
	}
	return *std::max_element(lengths.begin(), lengths.end());
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Return true if the schedule is feasible
/// </summary>
bool TaskGraph::checkSchedule() const {
	for (auto t1 : m_sorted) {
		const int end = t1->m_start + t1->m_duration;

		// check if each task is mapped exactly once
		size_t cnt = 0;
		for (auto& p : m_schedule) {
			cnt += p.count(t1);
		}
		if (cnt != 1) return false;

		// check for each task t1 if every dependent task t2 starts not before t1 ended
		for (auto t2 : t1->m_tasks) {
			if (t2->m_start < end) return false;
		}
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Return number of communication steps (send operations) in m_schedule
/// </summary>
int TaskGraph::communications() const {
	int sum = 0;

	// sum up the communications for each process
	for (auto pit = m_schedule.begin(); pit < m_schedule.end(); pit++) {
		sum += pit->communications();
	}
	return sum;
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Simple brute force search (DFS)
/// </summary>
/// <param name="idx">index of the current node</param>
/// <param name="bestSolution">current best duration</param>
/// <returns>best duration</returns>
int TaskGraph::searchSimple(int idx, int bestSolution) {
	if (idx == m_sorted.size()) {
		// solution found
		const int solution = duration(m_processes);

		if (solution < bestSolution) {
			// local minimimum found, store solution in m_schedule
#ifdef SHOW_PROGRESS
			std::cout << "Current best solution: " << solution << std::endl;
#endif
			m_schedule = m_processes;
		}
		return solution;

	} else {
		// recursive branch
		std::vector<int> startTimes(m_size - idx);
		Task* t = m_sorted[idx];

		// copy current task start times
		for (size_t i = idx; i < m_size; i++) {
			startTimes[i - idx] = m_sorted[i]->m_start;
		}

		// place t in all processes
		for (auto it = m_processes.begin(); it < m_processes.end(); it++) {
			Process& p = *it;
			const int pDur = p.getDuration();
			int duration = t->m_duration;

			if (pDur < t->m_start) {
				// task t starts after a delay
				const int delay = t->m_start - pDur;
				duration += delay;
			} else if (t->m_start < pDur) {
				// update start times
				t->updateStart(pDur);
			}

			// new process duration is less than upper bound -> update process
			p += duration;
			p += t;

			// recursive call
			const int dur = searchSimple(idx + 1, bestSolution);
			if (dur < bestSolution) {
				bestSolution = dur;
			}

			// remove task t from process p
			p -= duration;
			p -= t;

			// restore task start times
			for (size_t i = idx; i < m_size; i++) {
				m_sorted[i]->m_start = startTimes[i - idx];
			}
		}
		return bestSolution;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Branch and bound with early return (DFS)
/// </summary>
/// <param name="idx">index of the current node</param>
/// <param name="bestSolution">current best duration</param>
/// <returns>best duration</returns>
int TaskGraph::searchBranchAndBound(int idx, int bestSolution) {
	// TODO
	if (idx == m_sorted.size()) {
		// solution found
		const int solution = duration(m_processes);

		if (solution < bestSolution) {
			// local minimimum found, store solution in m_schedule
#ifdef SHOW_PROGRESS
			std::cout << "Current best solution: " << solution << std::endl;
#endif
			m_schedule = m_processes;
		}
		return solution;

	} else {
		// recursive branch
		std::vector<int> startTimes(m_size - idx);
		Task* t = m_sorted[idx];

		// copy current task start times
		for (size_t i = idx; i < m_size; i++) {
			startTimes[i - idx] = m_sorted[i]->m_start;
		}

		// place t in some processes if the resulting duration is less than the current best solution
		for (auto it = m_processes.begin(); it < m_processes.end(); it++) {
			Process& p = *it;
			const int pDur = p.getDuration();
			int duration = t->m_duration;

			if (pDur < t->m_start) {
				// task t starts after a delay
				const int delay = t->m_start - pDur;
				duration += delay;
			}
			if (pDur + duration < bestSolution) {
				if (t->m_start < pDur) {
					// update start times
					t->updateStart(pDur);
				}

				// new process duration is less than upper bound -> update process
				p += duration;
				p += t;

				// recursive call
				const int dur = searchBranchAndBound(idx + 1, bestSolution);
				if (dur < bestSolution) {
					bestSolution = dur;
					if (dur == m_criticalPathLen) return dur; // early return
				}

				// remove task t from process p
				p -= duration;
				p -= t;
			}

			// restore task start times
			for (size_t i = idx; i < m_size; i++) {
				m_sorted[i]->m_start = startTimes[i - idx];
			}
		}
		return bestSolution;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Parallel branch and bound with early return and process selection (DFS)
/// </summary>
/// <param name="idx">index of the current node</param>
/// <param name="bestSolution">current best duration</param>
/// <returns>best duration</returns>
int TaskGraph::searchParallelDFS(int idx, int bestSolution) {
	const int nThreads = std::thread::hardware_concurrency(); // parallel threads
	std::cout << "Number of threads: " << nThreads << std::endl;
	std::vector<std::future<int>> futures(nThreads);
	std::vector<DFSearcher> searchers(nThreads, DFSearcher(*this));
	int id = 0;

	// add work to first searcher
	searchers[0].addInitalWork(m_processes.size(), idx);

	// start parallel searching
	for (auto& f : futures) {
		DFSearcher& s = searchers[id];

		s.setID(id);
		s.setSearchers(&searchers);

		f = std::async(std::launch::async, [&s, bestSolution] {
			return s.startSearching(bestSolution);
		});
		id++;
	}

	// reduce results
	id = 0;
	for (auto& f : futures) {
		const int sol = f.get();

		if (sol < bestSolution) {
			bestSolution = sol;
			m_schedule = searchers[id].getSchedule();
		}
		id++;
	}
	return bestSolution;
}

////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Find the best task mapping (schedule)
/// </summary>
/// <param name="root">root node of the task graph</param>
/// <param name="searchAlgorithm">a search algorithm</param>
/// <returns>duration of task schedule</returns>
int TaskGraph::findMapping(Task* root, int searchAlgorithm) {
	assert(1 <= searchAlgorithm && searchAlgorithm <= 5);
	topSort(root);
	m_criticalPathLen = criticalPathLength();
	std::cout << "Critical Path Length: " << m_criticalPathLen << std::endl << std::endl;

	int duration = 0;
	switch (searchAlgorithm) {
	case 1: duration = searchSimple(0, INT32_MAX); break;
	case 2: duration = searchBranchAndBound(0, INT32_MAX); break;
	case 3: duration = searchParallelDFS(0, INT32_MAX); break;
	default: assert(false);
	}
	std::cout << "Duration: " << duration << std::endl;
	std::cout << "Communications (send): " << communications() << std::endl;

	// show schedule
	std::cout << "\nSchedule" << std::endl;
	for (size_t i = 0; i < m_schedule.size(); i++) {
		const Process& p = m_schedule[i];

		std::cout << "p" << i << ": " << p << std::endl;
	}
	std::cout << "is valid: " << std::boolalpha << checkSchedule() << std::endl;

	return duration;
}

