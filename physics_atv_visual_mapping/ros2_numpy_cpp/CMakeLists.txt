cmake_minimum_required(VERSION 3.8)
project(ros2_numpy_cpp)

cmake_policy(SET CMP0023 OLD)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(pybind11 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(perception_interfaces REQUIRED)


find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
include_directories(${PYBIND11_INCLUDE_DIRS})

pybind11_add_module(ros2_numpy_cpp src/grid_pub.cpp)
ament_target_dependencies(ros2_numpy_cpp 
  rclcpp 
  sensor_msgs
  perception_interfaces
)