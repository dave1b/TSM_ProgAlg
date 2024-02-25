#include "DFSearcher.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>

constexpr int MinSplitSize = 2;

////////////////////////////////////////////////////////////////////////////////////////
int DFSearcher::startSearching(int bestSolution) {
	m_bestSolution = bestSolution;

	while (m_searching) {
		Node v = remove();

		if (v.isValid()) {
			const int idx = v.m_idx; // current task index
			
			if (idx == m_sorted.size()) {
				// solution found
				const int solution = duration(v.m_processes);

				if (solution < m_bestSolution) {
					// local minimimum found, store solution in m_schedule
					m_bestSolution = solution;

#ifdef SHOW_PROGRESS
					std::cout << "Current best solution: " << m_bestSolution << " of searcher: " << m_myID << std::endl;
#endif
					m_schedule = v.m_processes;

					if (m_bestSolution == m_criticalPathLen) {
						// inform other threads about found optimum -> stop searching
						broadcastStop();
					} else {
						// inform other threads about new upper bound
						broadcastBestSolution();
					}
				}

			} else {
				Task* t = m_sorted[idx]; // t is the current task
				size_t i = 0;

				// place t in a subset of all processes
				for (auto it = v.m_processes.begin(); it < v.m_processes.end(); it++, i++) {
					const int pDur = it->getDuration();
					int duration = t->m_duration;

					// check for duplicate duration
					if (it == std::find_if(v.m_processes.begin(), it, [pDur](const Process& p) {
						return p.getDuration() == pDur;
					})) {
						// no duplicate duration found
						const int tStart = v.m_startTimes[t->m_idx];

						if (pDur < tStart) {
							// task t starts after a delay
							const int delay = tStart - pDur;
							duration += delay;
						}
						if (pDur + duration < m_bestSolution) {
							Node v2(v);
							Process& p = v2.m_processes[i];

							if (t->m_start < pDur) {
								// update start times
								v2.updateStartTimes(m_sorted, idx, pDur);
							}

							// new process duration is less than upper bound -> update process
							p += duration;
							p += t;

							// add new work to open list
							v2.m_idx++;
							add(v2);

							// remove task t from process p
							p -= duration;
							p -= t;
						}
					}
				}
			}
		} else {
			// work stealing
			size_t max = 0;
			int id = 0, i = 0;

			m_outOfWork = true;

			{
				// find searcher with largest amount of work
				for (auto& s : *m_searchers) {
					size_t openNodes = s.openNodes();

					if (s.m_myID != m_myID && openNodes > max) {
						max = openNodes;
						id = i;
					}
					i++;
				}

				if (max >= MinSplitSize && id != m_myID) {
					auto& s = (*m_searchers)[id];

					//std::cout << "Work stealing" << std::endl;
					if (m_searching) stealWorkFrom(s);
					//std::cout << openNodes() << std::endl;
				}
			}
			
			sleep(); 
			checkForEnd();
		}
	}

	return m_bestSolution;
}