#include "TaskGraph.h"
#include "Stopwatch.h"
#include <iostream>
#include <array>
#include <algorithm>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////
static void findMapping(TaskGraph& g, Task* root, int s) {
	Stopwatch sw;

	sw.Start();
	g.findMapping(root, s);
	sw.Stop();

	cout << "\nElapsed Time [s]: " << sw.GetElapsedTimeSeconds() << endl;
}

////////////////////////////////////////////////////////////////////////////////////////
static void graph1(int p, int s) {
	cout << "\n\nGraph 1 (Ex. 3.2 a)" << endl;

	Task t1(1, 5);
	Task t2(2, 6);
	Task t3(3, 2);
	Task t4(4, 1);
	Task t5(5, 4);
	Task t6(6, 5);
	Task t7(7, 1);
	Task t8(8, 3);
	Task t9(9, 5);
	Task t10(10, 7);
	Task t11(11, 2);
	Task t12(12, 8);
	Task t13(13, 4);
	Task t14(14, 9);
	Task t15(15, 3);

	t1.addTasks({ &t2, &t3 });
	t2.addTasks({ &t4, &t5 });
	t3.addTasks({ &t6, &t7 });
	t4.addTasks({ &t8, &t9 });
	t5.addTasks({ &t10, &t11 });
	t6.addTasks({ &t12, &t13 });
	t7.addTasks({ &t14, &t15 });

	TaskGraph g(p, 15);

	findMapping(g, &t1, s);
}

////////////////////////////////////////////////////////////////////////////////////////
static void graph2(int p, int s) {
	cout << "\n\nGraph 2 (Ex. 3.2 b)" << endl;

	Task t0(0, 0); // additional root node with duration 0
	Task t1(1, 1);
	Task t2(2, 3);
	Task t3(3, 4);
	Task t4(4, 7);
	Task t5(5, 9);
	Task t6(6, 1);
	Task t7(7, 2);
	Task t8(8, 5);
	Task t9(9, 5);
	Task t10(10, 5);
	Task t11(11, 2);
	Task t12(12, 4);
	Task t13(13, 3);
	Task t14(14, 3);
	Task t15(15, 2);

	t0.addTasks({ &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8 });
	t1.addTasks({ &t9 });
	t2.addTasks({ &t9 });
	t3.addTasks({ &t10 });
	t4.addTasks({ &t10 });
	t5.addTasks({ &t11 });
	t6.addTasks({ &t11 });
	t7.addTasks({ &t12 });
	t8.addTasks({ &t12 });
	t9.addTasks({ &t13 });
	t10.addTasks({ &t13 });
	t11.addTasks({ &t14 });
	t12.addTasks({ &t14 });
	t13.addTasks({ &t15 });
	t14.addTasks({ &t15 });

	TaskGraph g(p, 16);

	findMapping(g, &t0, s);
}

////////////////////////////////////////////////////////////////////////////////////////
static void graph3(int p, int s) {
	cout << "\n\nGraph 3 (Ex. 3.3 LU Decomposition)" << endl;

	// TODO create graph and find mapping

}

////////////////////////////////////////////////////////////////////////////////////////
int main() {
	int p = 1;
	int s = 1;
	int g = 0;

	cout << "Graphs (1 - 3): "; cin >> g;
	cout << "Number of processes: "; cin >> p;
	cout << "Search algorithm (1: simple, 2: fast, 3: parallelDFS): "; cin >> s;

	switch (g) {
	case 0: graph1(p, s); graph2(p, s); graph3(p, s); break;
	case 1: graph1(p, s); break;
	case 2: graph2(p, s); break;
	case 3: graph3(p, s); break;
	default: assert(false);
	}

}