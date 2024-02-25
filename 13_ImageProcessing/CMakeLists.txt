# CMakeList.txt : CMake project for ProgAlg, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Set target name
set(TARGET_NAME 13-ImageProcessing)

# Set tests directory
set(TESTS_DIR "${PROJECT_SOURCE_DIR}/Tests")

# Search modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(FreeImage REQUIRED)
find_package(FreeImagePlus REQUIRED)
find_package(SYCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

# Set source files (h-files are optional)
set(SOURCE_FILES "main.cpp" "imageprocessing.cpp")

# Add source to this project's executable.
add_executable(${TARGET_NAME} ${SOURCE_FILES})

# Add additional include directory
target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/Stopwatch")
#target_include_directories(${TARGET_NAME} PRIVATE "${FREEIMAGEPLUS_INCLUDE_PATH}")

# Add library for parallel execution
target_link_libraries(${TARGET_NAME} PRIVATE tbb "${FREEIMAGEPLUS_LIBRARIES}" "${FREEIMAGE_LIBRARIES}")

# Set compiler and linker options
if(CUDA_VERSION_MAJOR)
    message(STATUS "CUDA available")
    set(DPCPP_FLAGS ${DPCPP_FLAGS} -qopenmp -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else()
    message(STATUS "CUDA not found")
    set(DPCPP_FLAGS ${DPCPP_FLAGS} -qopenmp -fsycl)
endif()
add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SOURCE_FILES})

# copy DLLs to output folder
if(WIN32)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD     # Adds a post-build event to ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different       # which executes "cmake - E copy_if_different..."
            "${FREEIMAGE_LIB_PATH}/FreeImage.dll"           # <--this is in-file
            $<TARGET_FILE_DIR:${TARGET_NAME}>)              # <--this is out-file path
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD     # Adds a post-build event to ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different       # which executes "cmake - E copy_if_different..."
            "${FREEIMAGEPLUS_LIB_PATH}/FreeImagePlus.dll"   # <--this is in-file
            $<TARGET_FILE_DIR:${TARGET_NAME}>)              # <--this is out-file path
endif()

# copy image.png to output folder
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD     # Adds a post-build event to ${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different       # which executes "cmake - E copy_if_different..."
        "${TESTS_DIR}/image.png"                        # <--this is in-file
        $<TARGET_FILE_DIR:${TARGET_NAME}>)              # <--this is out-file path
