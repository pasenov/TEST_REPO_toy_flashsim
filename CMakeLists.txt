cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(toy_flashsim)

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
## find_package(xtensor REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(toy_flashsim toy_flashsim.cpp)

## target_include_directories(toy_flashsim PUBLIC ${xtensor_INCLUDE_DIRS})
## target_link_libraries(toy_flashsim PUBLIC xtensor ${TORCH_LIBRARIES})

target_link_libraries(toy_flashsim ${TORCH_LIBRARIES})

set_property(TARGET toy_flashsim PROPERTY CXX_STANDARD 17)

