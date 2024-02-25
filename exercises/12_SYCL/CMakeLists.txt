# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 12-SYCL)

# Search modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(SYCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

# Set source files (h-files are optional)
set(SOURCE_FILES "main.cpp")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/Stopwatch")

# Add library for parallel execution
target_link_libraries(${TARGET_NAME} tbb)

# Set compiler and linker options
if(CUDA_VERSION_MAJOR)
    message(STATUS "CUDA available")
    set(DPCPP_FLAGS ${DPCPP_FLAGS} -qopenmp -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else()
    message(STATUS "CUDA not found")
    set(DPCPP_FLAGS ${DPCPP_FLAGS} -qopenmp -fsycl)
endif()
add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SOURCE_FILES})