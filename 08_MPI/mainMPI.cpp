#include <iostream>
#include <string>
#include <array>
#include "mpi.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
void receive(int id, int p)
{

	std::array<char, 50> greeting;
	p = 2;

	std::cout << "Process " << id << " receives greetings from " << p - 1 << " processes!" << std::endl;
	for (int i = 1; i < p; i++)
	{
		MPI_Recv(greeting.data(), (int)greeting.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	std::cout << "Greetings, done!" << std::endl;
}
void send(int id)
{
	std::cout << "Send greetings from process " + std::to_string(id) << std::endl;
	std::string greeting("Greetings from process " + std::to_string(id));
	MPI_Send(greeting.data(), (int)greeting.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
	int p, id;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	receive(id, p);
	send(id);

	MPI_Finalize();
}
