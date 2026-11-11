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

// Relays odometry with the pose covariance set from parameters, optionally anchored (tighter)
// for the first meters flown, so loop closures do not drag the start.

#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"

namespace sslam
{

class OdomCovarianceNode : public rclcpp::Node
{
public:
  OdomCovarianceNode()
  : Node("odom_covariance_node")
  {
    const std::string input_topic = declare_parameter<std::string>("input_topic", "/odom");
    const std::string output_topic = declare_parameter<std::string>("output_topic", "/odom_cov");
    const double position_std = declare_parameter<double>("position_std_m", 0.0);  // x, y, z
    const double attitude_std = declare_parameter<double>("attitude_std_deg", 0.0);  // roll, pitch, yaw
    anchor_distance_ = declare_parameter<double>("anchor_distance_m", 0.0);  // m flown, 0 disables
    const double anchor_position_std = declare_parameter<double>("anchor_position_std_m", 0.005);
    const double anchor_attitude_std = declare_parameter<double>("anchor_attitude_std_deg", 0.0);  // <= 0: attitude_std_deg

    position_var_ = position_std * position_std;
    attitude_var_ = std::pow(attitude_std * M_PI / 180.0, 2);
    override_position_ = position_std > 0.0;
    override_attitude_ = attitude_std > 0.0;
    anchor_position_var_ = anchor_position_std * anchor_position_std;
    anchor_attitude_var_ = anchor_attitude_std > 0.0 ? std::pow(anchor_attitude_std * M_PI / 180.0, 2) : attitude_var_;

    pub_ = create_publisher<nav_msgs::msg::Odometry>(output_topic, rclcpp::SensorDataQoS());
    sub_ = create_subscription<nav_msgs::msg::Odometry>(input_topic, rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::UniquePtr msg) {callback(std::move(msg));});

    char buf[512];
    std::string pos = "as received", att = "as received", anchor;
    if (override_position_) {std::snprintf(buf, sizeof(buf), "%.4f m", std::sqrt(position_var_)); pos = buf;}
    if (override_attitude_) {std::snprintf(buf, sizeof(buf), "%.4f deg", std::sqrt(attitude_var_) * 180.0 / M_PI); att = buf;}
    if (anchor_distance_ > 0.0) {
      std::snprintf(buf, sizeof(buf), ", anchored (%.4f m, %.4f deg) for the first %.1f m",
        std::sqrt(anchor_position_var_), std::sqrt(anchor_attitude_var_) * 180.0 / M_PI, anchor_distance_);
      anchor = buf;
    }
    RCLCPP_INFO(get_logger(), "'%s' -> '%s': position std %s, attitude std %s%s",
      input_topic.c_str(), output_topic.c_str(), pos.c_str(), att.c_str(), anchor.c_str());
  }

private:
  void callback(nav_msgs::msg::Odometry::UniquePtr msg)
  {
    const auto & p = msg->pose.pose.position;
    if (last_xy_) {
      travelled_ += std::hypot(p.x - last_xy_->first, p.y - last_xy_->second);
    }
    last_xy_ = std::make_pair(p.x, p.y);
    const bool anchored = travelled_ < anchor_distance_;
    const double position_var = anchored ? anchor_position_var_ : position_var_;
    const double attitude_var = anchored ? anchor_attitude_var_ : attitude_var_;

    auto & cov = msg->pose.covariance;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const bool oi = i < 3 ? override_position_ : override_attitude_;
        const bool oj = j < 3 ? override_position_ : override_attitude_;
        if (oi || oj) {cov[6 * i + j] = 0.0;}
      }
    }
    if (override_position_) {
      for (int i = 0; i < 3; ++i) {cov[7 * i] = position_var;}
    }
    if (override_attitude_) {
      for (int i = 3; i < 6; ++i) {cov[7 * i] = attitude_var;}
    }
    pub_->publish(std::move(msg));
  }

  double position_var_, attitude_var_, anchor_position_var_, anchor_attitude_var_;
  bool override_position_, override_attitude_;
  double anchor_distance_;
  double travelled_ = 0.0;
  std::optional<std::pair<double, double>> last_xy_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
};

}  // namespace sslam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sslam::OdomCovarianceNode>());
  rclcpp::shutdown();
  return 0;
}
