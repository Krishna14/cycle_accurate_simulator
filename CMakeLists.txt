cmake_minimum_required(VERSION 3.16)

project(accel_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(accel_sim accel_sim.cpp)

target_compile_options(accel_sim PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)


