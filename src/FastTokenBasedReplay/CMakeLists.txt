cmake_minimum_required(VERSION 3.14)
project(MyLibrary)

set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

# Find pybind11
find_package(pybind11 REQUIRED)

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
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable
file(GLOB TEST_SOURCES tests/*.cpp)
add_executable(run_tests ${TEST_SOURCES})

# Link against GoogleTest
target_link_libraries(run_tests PRIVATE ${GTEST_BOTH_LIBRARIES} pthread)

# Register tests
add_test(NAME MyTests COMMAND run_tests)

# Custom target to build and run tests
add_custom_target(run_tests_target
    COMMAND run_tests
    DEPENDS run_tests
    COMMENT "Running tests..."
)

add_dependencies(run_tests_target run_tests)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O2") # Enable profiling

