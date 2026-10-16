cmake_minimum_required(VERSION 3.15...3.31)
project(FastTokenBasedReplay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FTR_BUILD_PYTHON "Build the Python extension module" ON)
option(FTR_BUILD_TESTS "Build the C++ unit tests (uses GoogleTest)" OFF)

# Python extension module, installed as FastTokenBasedReplay/_core
if(FTR_BUILD_PYTHON)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core MODULE src/bindings.cpp)
    target_include_directories(_core PRIVATE src)
    install(TARGETS _core DESTINATION FastTokenBasedReplay)
endif()

# C++ unit tests: cmake -S . -B build -DFTR_BUILD_TESTS=ON -DFTR_BUILD_PYTHON=OFF
if(FTR_BUILD_TESTS)
    enable_testing()

    # Use an installed GoogleTest if there is one, otherwise download it
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(run_tests tests/main.cpp)
    target_include_directories(run_tests PRIVATE src tests)
    target_link_libraries(run_tests PRIVATE GTest::gtest)

    include(GoogleTest)
    gtest_discover_tests(run_tests)

    # Small standalone benchmark
    add_executable(main src/main.cpp)
    target_include_directories(main PRIVATE src)
endif()
