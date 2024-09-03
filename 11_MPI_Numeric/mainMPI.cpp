#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////
// function declarations
void matrixMultiplicationTests();
void cannonsTests();
void integrationTests();

//////////////////////////////////////////////////////////////////////////////////////////////
// Matrix multiplaction tests
// For OpenMP performance measurements don't use mpirun, just run the program standalone
int main() {
	matrixMultiplicationTests();

	MPI_Init(nullptr, nullptr);

	cannonsTests();
	integrationTests();

	MPI_Finalize();
}