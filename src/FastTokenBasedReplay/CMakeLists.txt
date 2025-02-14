cmake_minimum_required(VERSION 3.15)
project(FastTokenBasedReplay)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# Set output directory to src/
set(LIB_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/..")  # Moves .so file to src/

pybind11_add_module(FastTokenBasedReplay bindings.cpp)

# Ensure the library is placed in src/
set_target_properties(FastTokenBasedReplay PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${LIB_OUTPUT_DIR}")

