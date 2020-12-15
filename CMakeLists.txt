cmake_minimum_required(VERSION 3.12)
project(D3T)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(DEBUG=1)
set (CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pthread -I/usr/local/include -L/usr/local/lib -lprotobuf")

add_library(
        sources SHARED
        third_party/protobuf/dataset.pb.h third_party/protobuf/dataset.pb.cc
        cpp/utils.h cpp/split.h cpp/noise.h cpp/entity.h cpp/coordinator.h
        cpp/run_helpers.h
)

add_executable(single_run cpp/single_run.cpp)
target_link_libraries(single_run PUBLIC protobuf sources)