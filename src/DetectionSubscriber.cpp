#include <string>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int16.hpp"
#include "image_transport/image_transport.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>

#include "darknet_ros_msgs/msg/mno.hpp"
#include "darknet_ros_msgs/msg/bounding_boxes.hpp"
#include "darknet_ros_msgs/msg/bounding_box.hpp"

#define RED_H_MAX 180
#define RED_H_MIN 155
#define RED_S_MAX 255
#define RED_S_MIN 70
#define RED_V_MAX 255
#define RED_V_MIN 0

#define BLUE_H_MAX 100
#define BLUE_H_MIN 70
#define BLUE_S_MAX 255
#define BLUE_S_MIN 50
#define BLUE_V_MAX 255
#define BLUE_V_MIN 50
#define THRESHOLD_RATIO 0.05
#define DECISION_RATE 0.8
#define CHECK_COUNT 15

using std::placeholders::_1;

int imageCount = 0;
int trafficLightCount = 0;
int redCount = 0;
int blueCount = 0;
int lightState = 0; // 1:赤、2：赤の後の青、0：その他
bool existTrafficLight = false;
bool isDetect_ = false;
std::vector<int> detectWayPoints;

class DetectionSubscriber : public rclcpp::Node
{
public:
  DetectionSubscriber()
      : Node("detection_subscriber")
  {
    subscription_mno = this->create_subscription<darknet_ros_msgs::msg::Mno>(
        std::string("/mno_topic"), 10, std::bind(&DetectionSubscriber::mnoCallback, this, _1));
    subscription_way = this->create_subscription<std_msgs::msg::Int16>(
        std::string("/current_waypoint_index"), 10, std::bind(&DetectionSubscriber::wayPointCallback, this, _1));
    publisher_ = this->create_publisher<std_msgs::msg::Bool>("is_runnable", 10);
    detectWayPoints.push_back(10);
    detectWayPoints.push_back(20);
  }

private:
  void mnoCallback(const darknet_ros_msgs::msg::Mno::SharedPtr msg) const
  {
    cv::Mat camImageMat, clipImage, hsvImage, maskImageRed, maskImageBlue;
    double whiteAreaRatio;
    if (isDetect_)
    {
      if (!existTrafficLight)
      {
        RCLCPP_INFO(get_logger(), "check for traffic light.");
        imageCount++;
        if (msg->is_detected)
          trafficLightCount++;
        if (imageCount >= CHECK_COUNT)
          existTrafficLight = filter_temporal();
      }
      else
      {
        if (msg->is_detected)
        {
          // RCLCPP_INFO(get_logger(), "check red or blue");
          trafficLightCount++;

          // 画像を取得して切り抜き
          cv_bridge::CvImagePtr cam_image = cv_bridge::toCvCopy(msg->image, "bgr8");
          camImageMat = cam_image->image.clone();
          int xmin, xmax, ymin, ymax;
          xmin = msg->bounding_boxes.bounding_boxes[0].xmin;
          xmax = msg->bounding_boxes.bounding_boxes[0].xmax;
          ymin = msg->bounding_boxes.bounding_boxes[0].ymin;
          ymax = msg->bounding_boxes.bounding_boxes[0].ymax;
          clipImage = cv::Mat(camImageMat, cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)).clone();
          // cv::namedWindow("clipImage");
          // cv::imshow("clipImage", clipImage);
          // cv::waitKey(10);

          // hsv画像に変換
          cv::cvtColor(clipImage, hsvImage, CV_BGR2HSV, 3);

          // 赤のフィルター
          maskImageRed = filter_color(hsvImage, RED_H_MAX, RED_H_MIN, RED_S_MAX, RED_S_MIN, RED_V_MAX, RED_V_MIN);
          // cv::namedWindow("maskImageRed");
          // cv::imshow("maskImageRed", maskImageRed);
          // cv::waitKey(1);

          whiteAreaRatio = calculate_white_ratio(maskImageRed);
          RCLCPP_INFO(get_logger(), "whiteAreaRatio red = %f", whiteAreaRatio);
          if (whiteAreaRatio > THRESHOLD_RATIO)
          {
            RCLCPP_INFO(get_logger(), "red light");
            redCount++;
          }

          // 青のフィルター
          maskImageBlue = filter_color(hsvImage, BLUE_H_MAX, BLUE_H_MIN, BLUE_S_MAX, BLUE_S_MIN, BLUE_V_MAX, BLUE_V_MIN);
          // cv::namedWindow("maskImageBlue");
          // cv::imshow("maskImageBlue", maskImageBlue);
          // cv::waitKey(1);
          whiteAreaRatio = calculate_white_ratio(maskImageBlue);
          RCLCPP_INFO(get_logger(), "whiteAreaRatio blue = %f", whiteAreaRatio);
          if (whiteAreaRatio > THRESHOLD_RATIO)
          {
            RCLCPP_INFO(get_logger(), "blue light");
            blueCount++;
          }

          if (trafficLightCount >= CHECK_COUNT)
          {
            determine_red_blue();
          }
        }

        if (lightState == 2)
        {
          std_msgs::msg::Bool msg = std_msgs::msg::Bool();
          msg.data = true;
          publisher_->publish(msg);
          RCLCPP_INFO(this->get_logger(), "publish true");
          imageCount = 0;
          trafficLightCount = 0;
          redCount = 0;
          blueCount = 0;
          lightState = 0;
          existTrafficLight = false;
          isDetect_ = false;
        }
        else
        {
          std_msgs::msg::Bool msg = std_msgs::msg::Bool();
          msg.data = false;
          publisher_->publish(msg);
          RCLCPP_INFO(this->get_logger(), "publish false");
        }
      }
    }
  }

  void wayPointCallback(const std_msgs::msg::Int16::SharedPtr msg) const
  {
    int wayPoint = msg->data;
    bool detectPointFlag = false;
    for (size_t i = 0; i < detectWayPoints.size(); i++)
    {
      if (wayPoint == detectWayPoints[i])
      {
        detectPointFlag = true;
        break;
      }
    }
    if (detectPointFlag) isDetect_ = true;
    else {
      if(isDetect_){
        imageCount = 0;
        trafficLightCount = 0;
        redCount = 0;
        blueCount = 0;
        lightState = 0;
        existTrafficLight = false;
        isDetect_ = false;
      }
      isDetect_ = false;
    }
  }

  bool filter_temporal() const
  {
    if ((double)trafficLightCount / (double)imageCount > 0.7)
    {
      trafficLightCount = 0;
      return true;
    }
    else
    {
      imageCount = 0;
      trafficLightCount = 0;
      return false;
    }
  }

  cv::Mat filter_color(cv::Mat src, int h_max, int h_min, int s_max, int s_min, int v_max, int v_min) const
  {
    cv::Mat result;
    cv::Scalar min = cv::Scalar(h_min, s_min, v_min);
    cv::Scalar max = cv::Scalar(h_max, s_max, v_max);
    cv::inRange(src, min, max, result);
    return result;
  }

  double calculate_white_ratio(cv::Mat src) const
  {
    int whole_area = src.size().width * src.size().height;
    int white_area = cv::countNonZero(src);
    return (double)white_area / (double)whole_area;
  }

  void determine_red_blue() const
  {
    double redRatio = (double)redCount / (double)trafficLightCount;
    if (redRatio > DECISION_RATE)
    {
      lightState = 1;
    }
    else
    {
      double blueRatio = (double)blueCount / (double)trafficLightCount;
      if (blueRatio > DECISION_RATE)
      {
        if (lightState == 1)
          lightState = 2;
      }
    }
    redCount = 0;
    blueCount = 0;
    trafficLightCount = 0;
  }

  rclcpp::Subscription<darknet_ros_msgs::msg::Mno>::SharedPtr subscription_mno;
  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr subscription_way;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionSubscriber>());
  rclcpp::shutdown();
  return 0;
}