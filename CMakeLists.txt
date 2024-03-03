cmake_minimum_required(VERSION 3.15...3.27)
project(LocalFischerDiscriminantAnalysis)

# Set the C++ standard to C++17 (nanobind needs this)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and require the development components
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

set(EIGEN_DIR "C:/cpp_libs/eigen-3.4.0" CACHE PATH "Path to the Eigen library")

include_directories(${EIGEN_DIR})

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/lfda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics.cpp
)

nanobind_add_module(lfda
    ${SOURCE_FILES}
)

target_link_libraries(lfda PRIVATE ${Python_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(lfda PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(lfda PRIVATE ${Python_INCLUDE_DIRS})
