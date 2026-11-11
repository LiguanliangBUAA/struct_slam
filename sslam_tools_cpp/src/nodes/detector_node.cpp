// Copyright 2026 Universidad Politecnica de Madrid (UPM).
//
// Author: Guanliang Li
// Contributor: Pedro Espinosa Angulo, Santiago Tapia Fernandez (supervised)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Detects walls in each BEV scan (/lidar_data) and publishes them (/detection_results).

#include <algorithm>
#include <cctype>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "msg_interfaces/msg/lidar_data.hpp"
#include "msg_interfaces/msg/objects.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sslam_tools_cpp/detectors/detector.hpp"
#include "sslam_tools_cpp/detectors/yolo_detector.hpp"
#include "sslam_tools_cpp/ros_params.hpp"

namespace sslam
{

class DetectorNode : public rclcpp::Node
{
public:
  DetectorNode()
  : Node("detector_node")
  {
    // Added to the scan's stamp before publishing. Downstream nodes look up the robot pose at
    // this stamp, so a negative value compensates scans stamped later than their geometry.
    stamp_offset_ = rclcpp::Duration::from_seconds(declare_parameter<double>("stamp_offset_sec", 0.0));

    std::string type = declare_parameter<std::string>("detector_type", "yolo");
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) {return std::tolower(c);});

    RosParamReader params(*this);
    detector_ = make_detector(type, params, ament_index_cpp::get_package_share_directory("sslam_tools_cpp"));

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("lidar_image", 10);
    detection_pub_ = create_publisher<msg_interfaces::msg::Objects>("detection_results", 10);
    sub_ = create_subscription<msg_interfaces::msg::LidarData>("lidar_data", 10,
        [this](msg_interfaces::msg::LidarData::ConstSharedPtr msg) {callback(*msg);});

    RCLCPP_INFO(get_logger(), "Using %s as Detector Node.", type.c_str());
    RCLCPP_INFO(get_logger(), "DETECTOR CONFIGURATION:\n%s", params.summary().c_str());
    if (const auto * yolo = dynamic_cast<const YoloDetector *>(detector_.get())) {
      RCLCPP_INFO(get_logger(), "YOLO backend: %s", yolo->backend_description().c_str());
      if (!yolo->backend_note().empty()) {
        RCLCPP_WARN(get_logger(), "%s", yolo->backend_note().c_str());
      }
    }
  }

private:
  void callback(const msg_interfaces::msg::LidarData & msg)
  {
    msg_interfaces::msg::Objects out;
    out.header = msg.header;
    out.header.stamp = (rclcpp::Time(msg.header.stamp) + stamp_offset_);
    out.method = detector_->name();

    const auto start = std::chrono::steady_clock::now();
    // mm -> cm
    std::vector<float> x(msg.x_data.size()), y(msg.y_data.size());
    std::transform(msg.x_data.begin(), msg.x_data.end(), x.begin(), [](float v) {return v / 10.0f;});
    std::transform(msg.y_data.begin(), msg.y_data.end(), y.begin(), [](float v) {return v / 10.0f;});

    SSCollection ssc = detector_->detect(x, y);
    out.endpoints = std::move(ssc.endpoints);
    out.columns_circles_centers = std::move(ssc.columns_circles_centers);
    out.columns_circles_radius = std::move(ssc.columns_circles_radius);
    out.others_circles_centers = std::move(ssc.others_circles_centers);
    out.others_circles_radius = std::move(ssc.others_circles_radius);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    detection_pub_->publish(out);

    if (detector_->config().capture_inference_image) {
      publish_image(detector_->results_image());
    }
    RCLCPP_INFO(get_logger(), "%s detection elapsed time: %.4f seconds", detector_->name().c_str(), elapsed);
  }

  void publish_image(const cv::Mat & bgr)
  {
    if (image_pub_->get_subscription_count() == 0) {
      return;
    }
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(640, 640));
    sensor_msgs::msg::Image img;
    img.header.stamp = now();
    img.header.frame_id = "lidar_frame";
    img.height = small.rows;
    img.width = small.cols;
    img.encoding = "bgr8";
    img.is_bigendian = 0;
    img.step = small.cols * 3;
    img.data.assign(small.data, small.data + small.total() * small.elemSize());
    image_pub_->publish(img);
  }

  std::unique_ptr<Detector> detector_;
  rclcpp::Duration stamp_offset_{0, 0};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<msg_interfaces::msg::Objects>::SharedPtr detection_pub_;
  rclcpp::Subscription<msg_interfaces::msg::LidarData>::SharedPtr sub_;
};

}  // namespace sslam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sslam::DetectorNode>());
  rclcpp::shutdown();
  return 0;
}
