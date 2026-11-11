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

// Fuses the wall detections of one scan (/detection_results) into one segment per wall,
// with the corners between them (/local_fused_map).

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

#include "msg_interfaces/msg/map_elementswith_distance.hpp"
#include "msg_interfaces/msg/objects.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sslam_tools_cpp/clustering.hpp"
#include "sslam_tools_cpp/geometry.hpp"
#include "sslam_tools_cpp/ros_params.hpp"

namespace sslam
{

struct LocalFusionConfig
{
  // DBSCAN distance terms
  double dist_threshold = 10.0;  // cm
  double angle_threshold_deg = 12.0;
  double overlap_thresh = 50.0;  // cm
  double eps = 0.8;
  int min_samples = 1;
  // Corner generation
  double wall_connection_threshold = 10.0;  // cm
  // Shorter walls are dropped
  double wall_length_threshold = 100.0;  // cm
  // Debug image
  bool publish_img_flag = true;
  int processed_img_size = 640;
  int base_img_size = 4096;
  int sensor_posx = 2048;
  int sensor_posy = 2048;

  void load(ParamReader & p)
  {
    dist_threshold = p.get_double("dist_threshold", dist_threshold);
    angle_threshold_deg = p.get_double("angle_threshold_deg", angle_threshold_deg);
    overlap_thresh = p.get_double("overlap_thresh", overlap_thresh);
    eps = p.get_double("eps", eps);
    min_samples = static_cast<int>(p.get_int("min_samples", min_samples));
    wall_connection_threshold = p.get_double("wall_connection_threshold", wall_connection_threshold);
    wall_length_threshold = p.get_double("wall_length_threshold", wall_length_threshold);
    publish_img_flag = p.get_bool("publish_img_flag", publish_img_flag);
    processed_img_size = static_cast<int>(p.get_int("processed_img_size", processed_img_size));
    base_img_size = static_cast<int>(p.get_int("base_img_size", base_img_size));
    sensor_posx = static_cast<int>(p.get_int("sensor_posx", sensor_posx));
    sensor_posy = static_cast<int>(p.get_int("sensor_posy", sensor_posy));
  }
};

class LocalFusionNode : public rclcpp::Node
{
public:
  LocalFusionNode()
  : Node("local_fusion_node")
  {
    RosParamReader params(*this);
    cfg_.load(params);
    RCLCPP_INFO(get_logger(), "LOCAL FUSION CONFIGURATION:\n%s", params.summary().c_str());
    angle_threshold_ = deg2rad(cfg_.angle_threshold_deg);

    map_pub_ = create_publisher<msg_interfaces::msg::MapElementswithDistance>("local_fused_map", 10);
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("fused_map_image", 10);
    sub_ = create_subscription<msg_interfaces::msg::Objects>("detection_results", 10,
        [this](msg_interfaces::msg::Objects::ConstSharedPtr msg) {callback(*msg);});
  }

private:
  void callback(const msg_interfaces::msg::Objects & msg)
  {
    RCLCPP_DEBUG(get_logger(), "Local Fusion callback triggered.");
    std::vector<Segment> endpoints;
    for (size_t i = 0; i + 3 < msg.endpoints.size(); i += 4) {
      endpoints.push_back({msg.endpoints[i], msg.endpoints[i + 1], msg.endpoints[i + 2], msg.endpoints[i + 3]});
    }
    if (endpoints.empty()) {
      RCLCPP_DEBUG(get_logger(), "No endpoints received for fusion.");
      return;
    }
    // LSD and Hough see both faces of a wall: skeletonize first
    if (msg.method == "lsd_detector" || msg.method == "hought_detector") {
      endpoints = merge_wall_envelopes(endpoints);
    }

    std::vector<Polar> polar;
    std::vector<Segment> valid;
    for (const auto & w : endpoints) {
      const Polar p = endpoints2polar(w);
      if (std::abs(p[3] - p[2]) > cfg_.wall_length_threshold) {
        polar.push_back(p);
        valid.push_back(w);
      }
    }

    // Cluster the segments and fuse each cluster into one wall
    std::vector<Polar> fused;  // d, alpha, d1, d2
    std::vector<std::array<double, 6>> fused_params;  // d, alpha, x1, y1, x2, y2
    if (!valid.empty()) {
      const std::vector<int> labels = dbscan_precomputed(distance_matrix(valid), cfg_.eps, cfg_.min_samples);
      std::map<int, std::vector<size_t>> clusters;  // ordered by label, as Python's set of labels
      for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != -1) {clusters[labels[i]].push_back(i);}
      }
      for (const auto & [label, members] : clusters) {
        (void)label;
        // Weighted by segment length: a short segment has a poorly known line
        double wsum = 0.0, dsum = 0.0, ssum = 0.0, csum = 0.0;
        for (size_t m : members) {
          const double w = std::abs(polar[m][3] - polar[m][2]);
          wsum += w;
          dsum += w * polar[m][0];
          ssum += w * std::sin(polar[m][1]);
          csum += w * std::cos(polar[m][1]);
        }
        const double mean_d = dsum / wsum;
        const double mean_alpha = pymod(std::atan2(ssum, csum), kTwoPi);
        const double p0x = mean_d * std::cos(mean_alpha), p0y = mean_d * std::sin(mean_alpha);
        const double dx = -std::sin(mean_alpha), dy = std::cos(mean_alpha);
        double pmin = std::numeric_limits<double>::infinity();
        double pmax = -std::numeric_limits<double>::infinity();
        for (size_t m : members) {
          const Segment e = polar2endpoints(polar[m]);
          for (int k = 0; k < 2; ++k) {
            const double proj = (e[2 * k] - p0x) * dx + (e[2 * k + 1] - p0y) * dy;
            pmin = std::min(pmin, proj);
            pmax = std::max(pmax, proj);
          }
        }
        const Polar wall = {mean_d, mean_alpha, pmin, pmax};
        fused.push_back(wall);
        const Segment seg = polar2endpoints(wall);
        fused_params.push_back({mean_d, mean_alpha, seg[0], seg[1], seg[2], seg[3]});
      }
    }

    msg_interfaces::msg::MapElementswithDistance out;
    out.header = msg.header;
    out.num_walls = static_cast<int32_t>(fused.size());
    for (const auto & w : fused) {
      out.wall_parameters.insert(out.wall_parameters.end(), w.begin(), w.end());
    }

    // Corners: intersections close to both segments
    for (size_t i = 0; i < fused_params.size(); ++i) {
      for (size_t j = i + 1; j < fused_params.size(); ++j) {
        const auto & w1 = fused_params[i];
        const auto & w2 = fused_params[j];
        const auto inter = calculate_line_intersection(w1[0], w1[1], w2[0], w2[1]);
        if (!inter) {continue;}
        const double d1 = distance_point_to_line_segment(*inter, Vec2(w1[2], w1[3]), Vec2(w1[4], w1[5]));
        const double d2 = distance_point_to_line_segment(*inter, Vec2(w2[2], w2[3]), Vec2(w2[4], w2[5]));
        if (d1 <= cfg_.wall_connection_threshold && d2 <= cfg_.wall_connection_threshold) {
          out.corners_points.push_back(static_cast<float>(inter->x()));
          out.corners_points.push_back(static_cast<float>(inter->y()));
        }
      }
    }
    out.num_corners = static_cast<int32_t>(out.corners_points.size() / 2);

    // Columns are passed through
    if (!msg.columns_circles_centers.empty()) {
      out.num_columns = static_cast<int32_t>(msg.columns_circles_radius.size());
      for (size_t i = 0; i < msg.columns_circles_radius.size(); ++i) {
        out.columns_xyr.push_back(msg.columns_circles_centers[2 * i]);
        out.columns_xyr.push_back(msg.columns_circles_centers[2 * i + 1]);
        out.columns_xyr.push_back(msg.columns_circles_radius[i]);
      }
    }

    map_pub_->publish(out);
    if (cfg_.publish_img_flag && image_pub_->get_subscription_count() > 0) {
      publish_debug_image(endpoints, out);
    }
  }

  Eigen::MatrixXd distance_matrix(const std::vector<Segment> & walls) const
  {
    const size_t n = walls.size();
    Eigen::MatrixXd dist = Eigen::MatrixXd::Zero(n, n);
    std::vector<double> incli(n);
    for (size_t i = 0; i < n; ++i) {
      const auto & w = walls[i];
      incli[i] = std::abs(w[2] - w[0]) > 1e-6 ? pymod(std::atan2(w[3] - w[1], w[2] - w[0]), kTwoPi) : kPi / 2;
    }
    const double dist_thresh = std::max(cfg_.dist_threshold, 1e-6);
    const double angle_thresh = std::max(angle_threshold_, 1e-6);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        const auto & a = walls[i];
        const auto & b = walls[j];
        const Vec2 P1(a[0], a[1]), P2(a[2], a[3]), Q1(b[0], b[1]), Q2(b[2], b[3]);
        const double delta_dis = std::min({distance_point_to_line(P1, Q1, Q2), distance_point_to_line(P2, Q1, Q2),
            distance_point_to_line(Q1, P1, P2), distance_point_to_line(Q2, P1, P2)});
        // As in the Python node: min(diff, pi - diff) with diff in [0, 2 pi), which goes
        // negative (reads as parallel) for diff > pi.
        const double diff = std::abs(incli[i] - incli[j]);
        const double delta_angle = std::min(diff, kPi - diff);
        const double d = std::max({delta_dis / dist_thresh, angle_logistic_function(delta_angle, angle_thresh),
            overlap_cal(a, b) / cfg_.overlap_thresh});
        dist(i, j) = d;
        dist(j, i) = d;
      }
    }
    return dist;
  }

  cv::Point to_img(double x, double y, double scale) const
  {
    return {static_cast<int>((x + cfg_.sensor_posx) * scale), static_cast<int>((cfg_.sensor_posy - y) * scale)};
  }

  void publish_debug_image(const std::vector<Segment> & raw, const msg_interfaces::msg::MapElementswithDistance & m)
  {
    const double scale = static_cast<double>(cfg_.processed_img_size) / cfg_.base_img_size;
    const int s = cfg_.processed_img_size;
    cv::Mat canvas(s, s, CV_8UC3, cv::Scalar(255, 255, 255));
    for (const auto & e : raw) {
      const cv::Point p1 = to_img(e[0], e[1], scale), p2 = to_img(e[2], e[3], scale);
      cv::line(canvas, p1, p2, cv::Scalar(255, 0, 0), 2);
      cv::putText(canvas, "W", (p1 + p2) / 2, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }
    for (size_t i = 0; i + 3 < m.wall_parameters.size(); i += 4) {
      const Segment e = polar2endpoints({m.wall_parameters[i], m.wall_parameters[i + 1],
            m.wall_parameters[i + 2], m.wall_parameters[i + 3]});
      const cv::Point p1 = to_img(e[0], e[1], scale), p2 = to_img(e[2], e[3], scale);
      cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 1);
      cv::putText(canvas, "F", (p1 + p2) / 2, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    }
    for (size_t i = 0; i + 1 < m.corners_points.size(); i += 2) {
      cv::circle(canvas, to_img(m.corners_points[i], m.corners_points[i + 1], scale), 5,
        cv::Scalar(0, 255, 255), -1);
    }
    for (size_t i = 0; i + 2 < m.columns_xyr.size(); i += 3) {
      cv::circle(canvas, to_img(m.columns_xyr[i], m.columns_xyr[i + 1], scale),
        static_cast<int>(m.columns_xyr[i + 2] * scale), cv::Scalar(0, 255, 0), 2);
    }
    sensor_msgs::msg::Image img;
    img.header.stamp = now();
    img.header.frame_id = "lidar_frame";
    img.height = s;
    img.width = s;
    img.encoding = "bgr8";
    img.step = s * 3;
    img.data.assign(canvas.data, canvas.data + canvas.total() * canvas.elemSize());
    image_pub_->publish(img);
  }

  LocalFusionConfig cfg_;
  double angle_threshold_;
  rclcpp::Publisher<msg_interfaces::msg::MapElementswithDistance>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<msg_interfaces::msg::Objects>::SharedPtr sub_;
};

}  // namespace sslam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sslam::LocalFusionNode>());
  rclcpp::shutdown();
  return 0;
}
