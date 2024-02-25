# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 09-MPI_Collective)

find_package(MPI REQUIRED)

# Set source files (h-files are optional)
set(SOURCE_FILES "mainMPI.cpp")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Add library for distributed execution
target_link_libraries(${TARGET_NAME} mpi)
