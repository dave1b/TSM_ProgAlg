# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 01-CPP)

# Set source files (h-files are optional)
set(SOURCE_FILES "main.cpp" "findmax.cpp" "rangequery.cpp" "summation.cpp" "checkresult.h")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/Stopwatch")

# Add library for parallel execution
# target_link_libraries(${TARGET_NAME} PRIVATE tbb)
target_link_libraries(${TARGET_NAME} PRIVATE ${TBB_IMPORTED_TARGETS})