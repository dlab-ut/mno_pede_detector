/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include "rclcpp/rclcpp.hpp"
#include "DetectionSubscriber.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mno_pede_detector::DetectionSubscriber>());
  rclcpp::shutdown();
  return 0;
}