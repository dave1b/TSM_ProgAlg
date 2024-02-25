#include <iostream>
#include <string>
#include <array>
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	int p, id;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (id == 0) {
		std::array<char, 50> greeting;

		std::cout << "Process " << id << " receives greetings from " << p - 1 << " processes!" << std::endl;
		for (int i = 1; i < p; i++) {
			// TODO use MPI_Recv
		}
		std::cout << "Greetings, done!" << std::endl;
	} else {
		// TODO use MPI_Send
	}

	MPI_Finalize();
}
