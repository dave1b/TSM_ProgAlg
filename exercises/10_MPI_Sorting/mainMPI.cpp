#include <ctime>
#include <cstdlib>
#include <string>
#include <iostream>
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
void oddevensort();
void shellsort();

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	srand((unsigned int)time(nullptr));

	MPI_Init(&argc, &argv);

	oddevensort();
	shellsort();

	MPI_Finalize();
}