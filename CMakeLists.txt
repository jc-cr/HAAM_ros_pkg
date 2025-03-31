cmake_minimum_required(VERSION 3.0.2)
project(haam_ros_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  actionlib
  cv_bridge
  image_transport
  std_msgs
  sensor_msgs
  roslib
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)



catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS actionlib cv_bridge  image_transport  std_msgs sensor_msgs 
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/HAAM_topic_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
