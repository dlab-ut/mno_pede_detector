cmake_minimum_required(VERSION 3.8)
project(mno_pede_detector)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(darknet_ros_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

set(dependencies
    rclcpp
    rclpy
    rclcpp_action
    std_msgs
    image_transport
    cv_bridge
    sensor_msgs
    geometry_msgs
    darknet_ros_msgs
    ament_index_cpp
)

add_executable(detection_subscriber src/DetectionSubscriber.cpp)
ament_target_dependencies(detection_subscriber ${dependencies})

install(TARGETS
  detection_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()