cmake_minimum_required(VERSION 3.20...4.3)
project(Feedforward_Neural_Network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.1
)

FetchContent_MakeAvailable(eigen)

add_executable(Feedforward_Neural_Network)

target_sources(Feedforward_Neural_Network
    PRIVATE
        src/main.cpp
)

target_include_directories(Feedforward_Neural_Network
    PRIVATE
        include
)

target_link_libraries(Feedforward_Neural_Network
    PRIVATE
        Eigen3::Eigen
)