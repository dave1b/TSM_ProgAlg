# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 04-OpenMP)

find_package(OpenMP REQUIRED)

# Set source files (h-files are optional)
set(SOURCE_FILES "main.cpp" "matrixRowSorting.cpp" "summation.cpp")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/Stopwatch")

# Add library for parallel execution
target_link_libraries(${TARGET_NAME} tbb)

# Set compiler and linker options
target_link_options(${TARGET_NAME} PRIVATE ${DPCPP_FLAGS} -qopenmp)