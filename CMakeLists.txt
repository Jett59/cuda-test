cmake_minimum_required(VERSION 3.28)

project(hello CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

add_executable(hello hello.cu)
