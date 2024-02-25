# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.12)

# set target name
set(TARGET_NAME ProgAlg)
project (${TARGET_NAME})

# specify C++ compiler
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED True)

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

# specify C++ flags
set(DPCPP_FLAGS -Wno-linker-warnings)

# Default C++ Flags for warnings and optimisation
set(WARNING_FLAGS "-Wall -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(EXTRA_W_FLAGS "-Wcast-align -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline -Wno-undef -Wno-unused -Wno-unused-command-line-argument")
set(OPT_FLAGS "-march=native -mtune=native -Ot -Ofast -fomit-frame-pointer")

# Adding the flags to the targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS} ${DISABLED_WARNINGS}")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS} -g -Og")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${DISABLED_WARNINGS} -g -Og")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)

# specify output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include sub-projects.
add_subdirectory(01_C++)
add_subdirectory(04_OpenMP)
add_subdirectory(05_Sorting)
add_subdirectory(06_TaskMapping)
add_subdirectory(08_MPI)
add_subdirectory(09_MPI_Collective)
add_subdirectory(10_MPI_Sorting)
add_subdirectory(11_MPI_Numeric)
add_subdirectory(12_SYCL)
add_subdirectory(13_ImageProcessing)

