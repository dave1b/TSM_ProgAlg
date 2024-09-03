#include <time.h>
#include <iostream>
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// num integration in the domain [0,1] of f(x) = 1/(1 + x*x)
// midpoint or rectangle rule
static double rectangleRule(int nIntervals)
{
	// TODO use MPI
	double pi = 0;
	int nProcs;
	int myID;

	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);

	// numerical integrate f
	const double w = 1.0 / nIntervals;
	const int blockLen = (nIntervals + nProcs - 1) / nProcs; // ceiling
	const int idx = myID * blockLen;

	double fsum = 0.0;

	// use block partitioning
	for (int i = idx; i < std::min(nIntervals, idx + blockLen); ++i)
	{
		const double x = (i + 0.5) * w;
		fsum += 1.0 / (1.0 + x * x);
	}

	// use cyclic partitioning
	/*for (int interval = id; interval < nIntervals; interval += p) {
		double x = w*(interval + 0.5);
		fsum += 1.0/(1.0 + x*x);
	}
	*/
	return w * fsum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// num integration in the domain [0,1] of f(x) = 1/(1 + x*x)
// trapezoidal rule
static double trapezoidalRule(int nIntervals)
{
	// TODO use MPI
	double pi = 0;
	int nProcs;
	int myID;

	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);

	// numerical integrate f
	const int blockLen = (nIntervals + nProcs - 1) / nProcs; // ceiling
	const double w = 1.0 / nIntervals;
	const int idx = myID * blockLen;

	double fsum = 0;

	if (idx < nIntervals)
	{
		const double x = idx * w;
		fsum = 1.0 / (1.0 + x * x);
		if (idx == 0)
			fsum /= 2.0;

		for (int i = idx + 1; i < std::min(nIntervals, idx + blockLen); ++i)
		{
			const double x = i * w;
			fsum += 1.0 / (1.0 + x * x);
		}
	}
	if (myID == nProcs - 1)
	{
		fsum += 0.25;
	}

	return w * fsum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void integrationTests()
{
	constexpr double ReferencePI = 3.141592653589793238462643; // ref value of pi for comparison
	int id, p;

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (id == 0)
	{
		std::cout << std::endl
				  << "Numerical integration" << std::endl;

		std::cout << "number of MPI processes = " << p << std::endl;
	}

	// numerical integration
	int nIntervals = 0;

	if (id == 0)
	{
		// if this is the Rank 0 node...
		std::cout << std::endl
				  << "Please enter the number of integration intervals: ";
		std::cin >> nIntervals;
		if (nIntervals <= 0)
		{
			std::cout << "NumIntervals must be greater than 0" << std::endl;
		}
	}

	// process 0 sends the number of intervals to all processes
	MPI_Bcast(&nIntervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (nIntervals > 0)
	{
		double pi1, pi2;

		const double start = MPI_Wtime();
		const double piRect = rectangleRule(nIntervals);
		MPI_Reduce(&piRect, &pi1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		const double inter = MPI_Wtime();
		const double piTrap = trapezoidalRule(nIntervals);
		MPI_Reduce(&piTrap, &pi2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double end = MPI_Wtime();

		if (id == 0)
		{
			std::cout << "rectangle rule  : pi = " << pi1 * 4 << ", delta = " << pi1 * 4 - ReferencePI << ", process 0 time [s] = " << inter - start << std::endl;
			std::cout << "trapezoidal rule: pi = " << pi2 * 4 << ", delta = " << pi2 * 4 - ReferencePI << ", process 0 time [s] = " << end - inter << std::endl;
		}
	}
}