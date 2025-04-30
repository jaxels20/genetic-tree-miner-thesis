cmake_minimum_required(VERSION 3.14)
project(MyLibrary)

set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

# Find pybind11
find_package(pybind11 REQUIRED)
# add_subdirectory(${CMAKE_SOURCE_DIR}/pybind11)

# Create a shared library for Python bindings
add_library(bindings MODULE src/bindings.cpp)
target_link_libraries(bindings PRIVATE pybind11::module)
set_target_properties(bindings PROPERTIES 
    PREFIX ""  
    OUTPUT_NAME "FastTokenBasedReplay"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..  
)

# Include the source directory for the tests
include_directories(src)

# Find GoogleTest (Manually specify location for Homebrew)
find_package(GTest REQUIRED)
# add_subdirectory(${CMAKE_SOURCE_DIR}/googletest)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable
file(GLOB TEST_SOURCES tests/*.cpp)
add_executable(run_tests ${TEST_SOURCES})

# Build the main.cpp file
add_executable(main src/main.cpp)

# Link against GoogleTest
target_link_libraries(run_tests PRIVATE ${GTEST_BOTH_LIBRARIES} pthread)
#target_link_libraries(run_tests PRIVATE gtest gtest_main pthread)

#set(CMAKE_BUILD_TYPE Debug) # Debug purposes

