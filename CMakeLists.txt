cmake_minimum_required(VERSION 3.13)
project(ocltest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -lOpenCL -msse4.1 -O3")
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_CXX_STANDARD 11)

add_executable(ocltest main.cpp)
