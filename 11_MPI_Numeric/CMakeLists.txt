# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 11-MPI_Numeric)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Set source files (h-files are optional)
set(SOURCE_FILES "mainMPI.cpp" "matrixmult.cpp" "Cannon.cpp" "Integration.cpp")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE ${MPI_CXX_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/Stopwatch")

# Add library for distributed execution
# target_link_libraries(${TARGET_NAME} PRIVATE mpi tbb)
target_link_libraries(${TARGET_NAME} PRIVATE ${TBB_IMPORTED_TARGETS})


# Set compiler and linker options
target_compile_options(${TARGET_NAME} PRIVATE ${DPCPP_FLAGS} -fopenmp)
target_link_options(${TARGET_NAME} PRIVATE ${DPCPP_FLAGS} -fopenmp)