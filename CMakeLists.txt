cmake_minimum_required(VERSION 3.18)

project(Fractalich VERSION 0.1.0 LANGUAGES CXX)

# ─── C++ configuration ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Fetch pybind11 automatically ──────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)

FetchContent_MakeAvailable(pybind11)

# ─── C++ core library ──────────────────────────────────────────────────────────
set(FRACTALICH_SOURCES
    src/fractal_metrics.cpp
    src/fractal_dimensions.cpp
    src/fractal_nn.cpp
)

add_library(fractalich_core STATIC
    ${FRACTALICH_SOURCES}
)

target_include_directories(fractalich_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ─── Python extension module ───────────────────────────────────────────────────
pybind11_add_module(
    fractalich_ext
    bindings/python_bindings.cpp
)

target_link_libraries(
    fractalich_ext
    PRIVATE
        fractalich_core
)

target_include_directories(
    fractalich_ext
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ─── Install Python extension ──────────────────────────────────────────────────
# The Python package is:
#
#     fractalich/
#         __init__.py
#         neural.py
#         ...
#
# Therefore fractalich_ext must be installed directly into
# the fractalich package directory, NOT fractalich/fractalich/.
install(
    TARGETS fractalich_ext
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
