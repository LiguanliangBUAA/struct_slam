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

// Associates the walls of each scan (/local_fused_map) with a global wall map built over the
// flight, and publishes the matched observations for the SLAM backend (/semantic_observations).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include "dps_slam_msgs/msg/detection_with_id_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "msg_interfaces/msg/map_elementswith_distance.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sslam_tools_cpp/assignment.hpp"
#include "sslam_tools_cpp/geometry.hpp"
#include "sslam_tools_cpp/global_map.hpp"
#include "sslam_tools_cpp/ros_params.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace sslam
{

constexpr double kInf = std::numeric_limits<double>::infinity();

struct GlobalFusionConfig
{
  // Association gates
  double match_dist_thresh = 30.0;  // cm
  double match_angle_thresh_deg = 10.0;
  double overlap_thresh = 20.0;  // cm
  double gate_threshold = 100.0;
  int max_misses_thresh = 3;
  double sensor_range = 3000.0;  // cm
  // Debug image
  bool publish_img_flag = true;
  int processed_img_size = 640;
  int base_img_size = 4096;
  int sensor_posx = 2048;
  int sensor_posy = 2048;
  double image_period_sec = 0.0;  // min wall-clock time between debug images (0 = every frame)
  // Movement compensation from TF
  bool movement_flag = true;
  std::string map_frame = "drone0/map";
  std::string lidar_frame = "drone0/base_link";
  bool manhattan_opt_flag = false;
  double corner_max_gap = 80.0;  // cm, corners shown in the debug image
  // Covariance declared for the observations sent to the SLAM backend
  double obs_dist_std_cm = 10.0;
  double obs_dist_std_per_m_cm = 0.0;
  double obs_angle_std_deg = 10.0;
  // Frames taken while turning faster than this are skipped, and for yaw_rate_settle_sec after
  double max_yaw_rate_deg_s = 0.0;
  double yaw_rate_settle_sec = 0.0;
  // Revisit association: gates widen with the m flown since the wall was last matched
  double revisit_dist_per_m_cm = 0.0;
  double revisit_angle_per_m_deg = 0.0;
  double revisit_max_extra_dist_cm = 30.0;
  double revisit_max_extra_angle_deg = 5.0;
  double revisit_slack_decay = 0.9;
  // Second cost < ratio * first: ambiguous, dropped (0 = disabled)
  double ambiguity_cost_ratio = 0.0;
  // Candidates that are one split wall (parallel, within this distance) are not ambiguous (0 = off)
  double ambiguity_same_line_cm = 0.0;
  // A local wall that lost the one-to-one assignment still updates its best admissible global wall
  bool multi_match_flag = false;
  // Frame consistency on revisits: fit one pose offset to all walls of the frame
  bool frame_consistency_flag = false;
  double consistency_inlier_dist_cm = 10.0;
  double consistency_trigger_slack_cm = 20.0;
  int consistency_min_inliers = 3;
  // Merge of global walls that grew into each other (older id kept)
  bool merge_flag = false;
  double merge_angle_thresh_deg = 5.0;
  double merge_dist_thresh = 15.0;  // cm, keep under match_dist_thresh
  double merge_gap_thresh = 100.0;  // cm
  int merge_min_hits = 5;
  // Observations of the kept wall also sent under the removed id, each with split information
  int merge_bridge_obs = 10;

  void load(ParamReader & p)
  {
    match_dist_thresh = p.get_double("match_dist_thresh", match_dist_thresh);
    match_angle_thresh_deg = p.get_double("match_angle_thresh_deg", match_angle_thresh_deg);
    overlap_thresh = p.get_double("overlap_thresh", overlap_thresh);
    gate_threshold = p.get_double("gate_threshold", gate_threshold);
    max_misses_thresh = static_cast<int>(p.get_int("max_misses_thresh", max_misses_thresh));
    sensor_range = p.get_double("sensor_range", sensor_range);
    publish_img_flag = p.get_bool("publish_img_flag", publish_img_flag);
    processed_img_size = static_cast<int>(p.get_int("processed_img_size", processed_img_size));
    base_img_size = static_cast<int>(p.get_int("base_img_size", base_img_size));
    sensor_posx = static_cast<int>(p.get_int("sensor_posx", sensor_posx));
    sensor_posy = static_cast<int>(p.get_int("sensor_posy", sensor_posy));
    image_period_sec = p.get_double("image_period_sec", image_period_sec);
    movement_flag = p.get_bool("movement_flag", movement_flag);
    map_frame = p.get_string("map_frame", map_frame);
    lidar_frame = p.get_string("lidar_frame", lidar_frame);
    manhattan_opt_flag = p.get_bool("manhattan_opt_flag", manhattan_opt_flag);
    corner_max_gap = p.get_double("corner_max_gap", corner_max_gap);
    obs_dist_std_cm = p.get_double("obs_dist_std_cm", obs_dist_std_cm);
    obs_dist_std_per_m_cm = p.get_double("obs_dist_std_per_m_cm", obs_dist_std_per_m_cm);
    obs_angle_std_deg = p.get_double("obs_angle_std_deg", obs_angle_std_deg);
    max_yaw_rate_deg_s = p.get_double("max_yaw_rate_deg_s", max_yaw_rate_deg_s);
    yaw_rate_settle_sec = p.get_double("yaw_rate_settle_sec", yaw_rate_settle_sec);
    revisit_dist_per_m_cm = p.get_double("revisit_dist_per_m_cm", revisit_dist_per_m_cm);
    revisit_angle_per_m_deg = p.get_double("revisit_angle_per_m_deg", revisit_angle_per_m_deg);
    revisit_max_extra_dist_cm = p.get_double("revisit_max_extra_dist_cm", revisit_max_extra_dist_cm);
    revisit_max_extra_angle_deg = p.get_double("revisit_max_extra_angle_deg", revisit_max_extra_angle_deg);
    revisit_slack_decay = p.get_double("revisit_slack_decay", revisit_slack_decay);
    ambiguity_cost_ratio = p.get_double("ambiguity_cost_ratio", ambiguity_cost_ratio);
    ambiguity_same_line_cm = p.get_double("ambiguity_same_line_cm", ambiguity_same_line_cm);
    multi_match_flag = p.get_bool("multi_match_flag", multi_match_flag);
    frame_consistency_flag = p.get_bool("frame_consistency_flag", frame_consistency_flag);
    consistency_inlier_dist_cm = p.get_double("consistency_inlier_dist_cm", consistency_inlier_dist_cm);
    consistency_trigger_slack_cm = p.get_double("consistency_trigger_slack_cm", consistency_trigger_slack_cm);
    consistency_min_inliers = static_cast<int>(p.get_int("consistency_min_inliers", consistency_min_inliers));
    merge_flag = p.get_bool("merge_flag", merge_flag);
    merge_angle_thresh_deg = p.get_double("merge_angle_thresh_deg", merge_angle_thresh_deg);
    merge_dist_thresh = p.get_double("merge_dist_thresh", merge_dist_thresh);
    merge_gap_thresh = p.get_double("merge_gap_thresh", merge_gap_thresh);
    merge_min_hits = static_cast<int>(p.get_int("merge_min_hits", merge_min_hits));
    merge_bridge_obs = static_cast<int>(p.get_int("merge_bridge_obs", merge_bridge_obs));
  }
};

// Distance terms between a wall and a reference wall (polar form, cm / rad)
struct WallDiff
{
  double dist;  // mean distance of the wall's endpoints to the reference line
  double angle;  // angle between the lines, [0, pi/2]
  double gap;  // along-wall gap on the reference direction (0 when they overlap)
};

WallDiff diff_calc(const Polar & new_wall, const Polar & g_wall)
{
  const Segment a = polar2endpoints(new_wall);
  const Segment b = polar2endpoints(g_wall);
  const Vec2 P1(a[0], a[1]), P2(a[2], a[3]), Q1(b[0], b[1]), Q2(b[2], b[3]);
  WallDiff d;
  d.dist = (distance_point_to_line(P1, Q1, Q2) + distance_point_to_line(P2, Q1, Q2)) / 2.0;
  const double incli_new = pymod(std::atan2(a[3] - a[1], a[2] - a[0]), kTwoPi);
  const double incli_g = pymod(std::atan2(b[3] - b[1], b[2] - b[0]), kTwoPi);
  double diff = pymod(std::abs(incli_new - incli_g), kPi);
  if (diff > kPi / 2) {
    diff = kPi - diff;
  }
  d.angle = diff;
  const Vec2 dir(-std::sin(g_wall[1]), std::cos(g_wall[1]));
  const double p1 = P1.dot(dir), p2 = P2.dot(dir), q1 = Q1.dot(dir), q2 = Q2.dot(dir);
  const double overlap = std::min(std::max(p1, p2), std::max(q1, q2)) - std::max(std::min(p1, p2), std::min(q1, q2));
  d.gap = overlap > 0 ? 0.0 : std::abs(overlap);
  return d;
}

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

struct WallObs
{
  int id;
  Polar rel_wall;  // in the lidar frame
  double cov_scale;
};

class GlobalFusionNode : public rclcpp::Node
{
public:
  GlobalFusionNode()
  : Node("global_fusion_node")
  {
    RosParamReader params(*this);
    cfg_.load(params);
    RCLCPP_INFO(get_logger(), "GLOBAL FUSION CONFIGURATION:\n%s", params.summary().c_str());
    match_angle_thresh_ = deg2rad(cfg_.match_angle_thresh_deg);
    merge_angle_thresh_ = deg2rad(cfg_.merge_angle_thresh_deg);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    // Own spin thread, so /tf keeps arriving while the callback waits in lookupTransform
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, true);

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("global_fused_map_image", 10);
    obs_pub_ = create_publisher<dps_slam_msgs::msg::DetectionWithIDArray>("semantic_observations", 10);
    sub_ = create_subscription<msg_interfaces::msg::MapElementswithDistance>("local_fused_map", 10,
        [this](msg_interfaces::msg::MapElementswithDistance::ConstSharedPtr msg) {callback(*msg);});
    RCLCPP_INFO(get_logger(), "Global Fusion Node initialized.");
  }

private:
  // ---------------------------------------------------------------- frame transform
  // Local walls (lidar frame) -> map frame at the current robot pose, rho >= 0, d1 <= d2
  std::vector<Polar> transform_walls(const std::vector<Polar> & walls) const
  {
    std::vector<Polar> out;
    out.reserve(walls.size());
    const double c = std::cos(robot_yaw_), s = std::sin(robot_yaw_);
    for (const auto & row : walls) {
      const Segment e = polar2endpoints(row);
      const double x1 = c * e[0] - s * e[1] + robot_x_;
      const double y1 = s * e[0] + c * e[1] + robot_y_;
      const double x2 = c * e[2] - s * e[3] + robot_x_;
      const double y2 = s * e[2] + c * e[3] + robot_y_;
      double theta = pymod(std::atan2(-(x2 - x1), y2 - y1), kTwoPi);
      double rho = x1 * std::cos(theta) + y1 * std::sin(theta);
      if (rho < 0) {
        rho = -rho;
        theta = pymod(theta + kPi, kTwoPi);
      }
      const double cx = rho * std::cos(theta), cy = rho * std::sin(theta);
      const Vec2 dir(-std::sin(theta), std::cos(theta));
      double d1 = Vec2(x1 - cx, y1 - cy).dot(dir);
      double d2 = Vec2(x2 - cx, y2 - cy).dot(dir);
      if (d1 > d2) {std::swap(d1, d2);}
      out.push_back({rho, theta, d1, d2});
    }
    return out;
  }

  std::vector<std::array<double, 3>> transform_columns(const std::vector<std::array<double, 3>> & cols) const
  {
    std::vector<std::array<double, 3>> out;
    const double c = std::cos(robot_yaw_), s = std::sin(robot_yaw_);
    for (const auto & r : cols) {
      out.push_back({c * r[0] - s * r[1] + robot_x_, s * r[0] + c * r[1] + robot_y_, r[2]});
    }
    return out;
  }

  // ---------------------------------------------------------------- association
  // Extra (distance [cm], angle [rad]) gate from the travel since the wall was last matched
  std::pair<double, double> revisit_slack(const GlobalWall & g) const
  {
    if (suppress_revisit_slack_) {
      return {0.0, 0.0};
    }
    return {std::min(cfg_.revisit_max_extra_dist_cm, cfg_.revisit_dist_per_m_cm * g.stale_m),
      deg2rad(std::min(cfg_.revisit_max_extra_angle_deg, cfg_.revisit_angle_per_m_deg * g.stale_m))};
  }

  double compute_cost(const Polar & local_w, const GlobalWall & g) const
  {
    // A wall seen from its other side is the other face of a partition, never this one
    if (!g.same_side(robot_x_, robot_y_)) {
      return kInf;
    }
    const WallDiff d = diff_calc(local_w, g.polar());
    const auto [extra_dist, extra_angle] = revisit_slack(g);
    const double dist_thresh = cfg_.match_dist_thresh + extra_dist;
    const double angle_thresh = match_angle_thresh_ + extra_angle;
    if (d.dist > dist_thresh || d.angle > angle_thresh) {
      return kInf;
    }
    // Along-wall gap is not widened (walls would merge across openings)
    if (d.gap > cfg_.overlap_thresh) {
      return kInf;
    }
    // Admission on the widened gates
    if (d.dist / dist_thresh + d.angle / angle_thresh >= cfg_.gate_threshold) {
      return kInf;
    }
    // Ranking on the base gates, so a stale wall does not look closer than it is
    return d.dist / cfg_.match_dist_thresh + d.angle / match_angle_thresh_;
  }

  std::vector<Polar> manhattan_pre_filter(const std::vector<Polar> & walls) const
  {
    if (!cfg_.manhattan_opt_flag || global_walls_.size() < 3) {
      return walls;
    }
    const double dom = optimizer_.find_dominant_direction(global_walls_);
    std::vector<Polar> out;
    for (const auto & w : walls) {
      const double global_theta = pymod(w[1] + robot_yaw_, kTwoPi);
      const double diff = pymod(global_theta - dom, kTwoPi);
      const double k = std::nearbyint(diff / (kPi / 2));
      const double target = pymod(dom + k * (kPi / 2), kTwoPi);
      if (std::abs(std::atan2(std::sin(global_theta - target), std::cos(global_theta - target))) < deg2rad(5.0)) {
        out.push_back(w);
      }
    }
    return out;
  }

  static std::optional<Vec2> corner_intersection(const Polar & w1, const Polar & w2, double max_gap)
  {
    const double denom = std::sin(w2[1] - w1[1]);
    if (std::abs(denom) < 1e-5) {
      return std::nullopt;
    }
    const double ix = (w1[0] * std::sin(w2[1]) - w2[0] * std::sin(w1[1])) / denom;
    const double iy = (w2[0] * std::cos(w1[1]) - w1[0] * std::cos(w2[1])) / denom;
    auto gap = [&](const Polar & w) {
        const double proj = -ix * std::sin(w[1]) + iy * std::cos(w[1]);
        const double lo = std::min(w[2], w[3]), hi = std::max(w[2], w[3]);
        if (proj < lo) {return lo - proj;}
        if (proj > hi) {return proj - hi;}
        return 0.0;
      };
    if (gap(w1) < max_gap && gap(w2) < max_gap) {
      return Vec2(ix, iy);
    }
    return std::nullopt;
  }

  enum class OffsetResult { kNone, kUnverified, kOffset };

  // Pose offset (dx, dy [cm], dyaw [rad]) aligning the frame with revisited walls.
  // kNone: no revisit in the frame. kUnverified: no single consistent offset.
  OffsetResult estimate_frame_offset(const std::vector<Polar> & local, Eigen::Vector3d & offset)
  {
    if (local.empty() || global_walls_.empty()) {
      return OffsetResult::kNone;
    }
    std::vector<Polar> pred = transform_walls(local);
    const double inlier = cfg_.consistency_inlier_dist_cm;

    // Candidate pairs within the widened gates (no overlap gate)
    std::vector<std::pair<int, const GlobalWall *>> pairs;
    std::set<int> revisit;
    for (size_t i = 0; i < pred.size(); ++i) {
      for (const auto & g : global_walls_) {
        if (!g.same_side(robot_x_, robot_y_)) {continue;}
        const auto [extra_dist, extra_angle] = revisit_slack(g);
        const WallDiff d = diff_calc(pred[i], g.polar());
        if (d.dist < cfg_.match_dist_thresh + extra_dist && d.angle < match_angle_thresh_ + extra_angle) {
          pairs.emplace_back(static_cast<int>(i), &g);
          if (extra_dist > cfg_.consistency_trigger_slack_cm) {
            revisit.insert(static_cast<int>(i));
          }
        }
      }
    }
    revisit_candidates_ = revisit;
    if (revisit.empty()) {
      return OffsetResult::kNone;
    }

    // Rotation: median angle difference
    std::vector<double> dth;
    for (const auto & [i, g] : pairs) {
      dth.push_back(pymod((g->theta - pred[i][1]) + kPi / 2, kPi) - kPi / 2);
    }
    const double dyaw = median(dth);
    const double saved_yaw = robot_yaw_;
    robot_yaw_ = pymod(robot_yaw_ + dyaw, kTwoPi);
    pred = transform_walls(local);
    robot_yaw_ = saved_yaw;

    // Per pair: normal constraint n . t = r, and along-wall extents for the overlap check
    const size_t np = pairs.size();
    std::vector<int> wall_idx(np);
    Eigen::MatrixXd normals(np, 2), dirs(np, 2), local_ext(np, 2), global_ext(np, 2);
    Eigen::VectorXd offsets(np);
    for (size_t k = 0; k < np; ++k) {
      const int i = pairs[k].first;
      const GlobalWall & g = *pairs[k].second;
      const Vec2 n(std::cos(g.theta), std::sin(g.theta));
      const Vec2 u(-std::sin(g.theta), std::cos(g.theta));
      const Segment e = polar2endpoints(pred[i]);
      const Segment ge = polar2endpoints(g.polar());
      wall_idx[k] = i;
      normals.row(k) = n.transpose();
      dirs.row(k) = u.transpose();
      offsets(k) = g.rho - n.dot(Vec2((e[0] + e[2]) / 2, (e[1] + e[3]) / 2));
      const double l1 = u.dot(Vec2(e[0], e[1])), l2 = u.dot(Vec2(e[2], e[3]));
      const double g1 = u.dot(Vec2(ge[0], ge[1])), g2 = u.dot(Vec2(ge[2], ge[3]));
      local_ext.row(k) << std::min(l1, l2), std::max(l1, l2);
      global_ext.row(k) << std::min(g1, g2), std::max(g1, g2);
    }

    // Number of distinct local walls agreeing with offset t, and which pairs agree
    auto support = [&](const Vec2 & t, std::vector<bool> & ok) {
        ok.assign(np, false);
        std::set<int> walls;
        const Eigen::VectorXd along = dirs * t;
        const Eigen::VectorXd normal_err = normals * t - offsets;
        for (size_t k = 0; k < np; ++k) {
          const double gap = std::max(global_ext(k, 0) - (local_ext(k, 1) + along(k)),
              (local_ext(k, 0) + along(k)) - global_ext(k, 1));
          if (std::abs(normal_err(k)) < inlier && gap < cfg_.overlap_thresh) {
            ok[k] = true;
            walls.insert(wall_idx[k]);
          }
        }
        return static_cast<int>(walls.size());
      };

    std::vector<Vec2> hypotheses = {Vec2::Zero()};
    for (size_t k = 0; k < np; ++k) {
      hypotheses.push_back(offsets(k) * normals.row(k).transpose());
    }
    for (size_t k = 0; k < np; ++k) {
      for (size_t l = k + 1; l < np; ++l) {
        Eigen::Matrix2d A;
        A.row(0) = normals.row(k);
        A.row(1) = normals.row(l);
        if (std::abs(A.determinant()) > 0.5) {  // > 30 deg apart
          hypotheses.push_back(A.partialPivLu().solve(Eigen::Vector2d(offsets(k), offsets(l))));
        }
      }
    }
    std::vector<std::pair<int, Vec2>> scored;
    std::vector<bool> ok;
    for (const auto & t : hypotheses) {
      scored.emplace_back(support(t, ok), t);
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) {return a.first > b.first;});
    const int best_n = scored[0].first;
    const Vec2 best_t = scored[0].second;
    const int zero_support = support(Vec2::Zero(), ok);
    if (best_n < cfg_.consistency_min_inliers) {
      return OffsetResult::kUnverified;
    }
    // Rival offset: as much support, different along the inliers' normals (e.g. one alcove over)
    std::vector<bool> best_ok;
    support(best_t, best_ok);
    std::vector<int> inl;
    for (size_t k = 0; k < np; ++k) {
      if (best_ok[k]) {inl.push_back(static_cast<int>(k));}
    }
    for (size_t s = 1; s < scored.size(); ++s) {
      if (scored[s].first < best_n) {continue;}
      double max_dev = 0.0;
      for (int k : inl) {
        max_dev = std::max(max_dev, std::abs(normals.row(k).dot(scored[s].second - best_t)));
      }
      if (max_dev > inlier) {
        return OffsetResult::kUnverified;
      }
    }
    if (best_n <= zero_support) {
      offset.setZero();  // pose already consistent
      return OffsetResult::kOffset;
    }
    // Least squares over the inliers; singular values under 0.3 * max are dropped (numpy's
    // lstsq rcond), leaving an unconstrained direction at 0
    Eigen::MatrixXd A(inl.size(), 2);
    Eigen::VectorXd b(inl.size());
    for (size_t r = 0; r < inl.size(); ++r) {
      A.row(r) = normals.row(inl[r]);
      b(r) = offsets(inl[r]);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    Eigen::VectorXd inv = Eigen::VectorXd::Zero(sv.size());
    for (int k = 0; k < sv.size(); ++k) {
      if (sv(k) > 0.3 * sv(0)) {inv(k) = 1.0 / sv(k);}
    }
    const Eigen::Vector2d t = svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose() * b;
    offset << t(0), t(1), dyaw;
    return OffsetResult::kOffset;
  }

  // Returns (local index, global id) matches and the unmatched local indices
  std::pair<std::vector<std::pair<int, int>>, std::vector<int>> associate(const std::vector<Polar> & local)
  {
    active_debug_corners_.clear();
    const int N = static_cast<int>(local.size());
    const int M = static_cast<int>(global_walls_.size());
    std::vector<int> all_local(N);
    std::iota(all_local.begin(), all_local.end(), 0);
    if (M == 0 || N == 0) {
      return {{}, all_local};
    }
    const std::vector<Polar> pred = transform_walls(local);
    Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(N + M, N + M, kInf);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        cost(i, j) = compute_cost(pred[i], global_walls_[j]);
      }
    }

    // Ambiguous observations
    std::set<int> ambiguous;
    const double ratio = cfg_.ambiguity_cost_ratio;
    if (ratio > 0.0 && M >= 2) {
      for (int i = 0; i < N; ++i) {
        std::vector<int> order(M);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return cost(i, a) < cost(i, b);});
        const double c0 = cost(i, order[0]), c1 = cost(i, order[1]);
        if (std::isfinite(c1) && c1 < ratio * std::max(c0, 0.1)) {
          if (cfg_.ambiguity_same_line_cm > 0.0) {
            const WallDiff d = diff_calc(global_walls_[order[0]].polar(), global_walls_[order[1]].polar());
            if (d.dist < cfg_.ambiguity_same_line_cm && d.angle < match_angle_thresh_) {
              continue;
            }
          }
          ambiguous.insert(i);
          cost.row(i).head(M).setConstant(kInf);
        }
      }
    }
    // Unverified revisit: drop the walls with a revisited candidate
    if (unverified_revisit_) {
      for (int i : revisit_candidates_) {
        ambiguous.insert(i);
        cost.row(i).head(M).setConstant(kInf);
      }
    }

    // Dummy nodes for unmatched local and global walls (high cost: take every admissible match)
    const double dummy = 1e3;
    for (int i = 0; i < N; ++i) {cost(i, M + i) = dummy;}
    for (int j = 0; j < M; ++j) {cost(N + j, j) = dummy;}
    cost.block(N, M, M, N).setZero();

    const std::vector<int> col = linear_sum_assignment(cost);
    std::vector<std::pair<int, int>> matched;  // local index, global index
    std::vector<int> unmatched;
    for (int r = 0; r < N; ++r) {
      const int c = col[r];
      if (c < M) {
        if (std::isfinite(cost(r, c))) {
          matched.emplace_back(r, c);
        } else {
          unmatched.push_back(r);
        }
      } else if (!ambiguous.count(r)) {
        unmatched.push_back(r);
      }
    }

    // Corners between matched walls (debug image)
    for (size_t a = 0; a < matched.size(); ++a) {
      for (size_t b = a + 1; b < matched.size(); ++b) {
        const GlobalWall & g1 = global_walls_[matched[a].second];
        const GlobalWall & g2 = global_walls_[matched[b].second];
        if (g1.hits < 2 || g2.hits < 2) {continue;}
        const Polar & l1 = local[matched[a].first];
        const Polar & l2 = local[matched[b].first];
        double angle_diff = pymod(std::abs(l1[1] - l2[1]), kPi);
        if (angle_diff > kPi / 2) {angle_diff = kPi - angle_diff;}
        if (angle_diff < deg2rad(30.0)) {continue;}
        if (!corner_intersection(l1, l2, cfg_.corner_max_gap)) {continue;}
        const auto global_pt = corner_intersection(g1.polar(), g2.polar(), cfg_.corner_max_gap);
        if (global_pt) {active_debug_corners_.push_back(*global_pt);}
      }
    }

    if (cfg_.multi_match_flag) {
      std::vector<int> still;
      for (int l : unmatched) {
        int best = 0;
        for (int j = 1; j < M; ++j) {
          if (cost(l, j) < cost(l, best)) {best = j;}
        }
        if (std::isfinite(cost(l, best))) {
          matched.emplace_back(l, best);
        } else {
          still.push_back(l);
        }
      }
      unmatched = still;
    }

    std::vector<std::pair<int, int>> final_matches;
    for (const auto & [l, g] : matched) {
      final_matches.emplace_back(l, global_walls_[g].id);
    }
    return {final_matches, unmatched};
  }

  // ---------------------------------------------------------------- map maintenance
  bool is_occluded(double tx, double ty, const std::vector<Polar> & walls,
    const std::vector<std::array<double, 3>> & cols) const
  {
    const double target_dist = std::hypot(tx - robot_x_, ty - robot_y_);
    auto ccw = [](const Vec2 & A, const Vec2 & B, const Vec2 & C) {
        return (C.y() - A.y()) * (B.x() - A.x()) > (B.y() - A.y()) * (C.x() - A.x());
      };
    const Vec2 A(robot_x_, robot_y_), B(tx, ty);
    for (const auto & w : walls) {
      const Segment e = polar2endpoints(w);
      const Vec2 C1(e[0], e[1]), C2(e[2], e[3]);
      if (ccw(A, C1, C2) != ccw(B, C1, C2) && ccw(A, B, C1) != ccw(A, B, C2)) {
        return true;
      }
    }
    for (const auto & c : cols) {
      // As in the Python node: the column's distance is taken from the origin, not the robot
      if (std::hypot(c[0], c[1]) > target_dist) {continue;}
      const double cross = std::abs((c[0] - robot_x_) * (ty - robot_y_) - (c[1] - robot_y_) * (tx - robot_x_));
      if (cross / target_dist < c[2] * 1.2) {
        return true;
      }
    }
    return false;
  }

  bool merge_candidate(const GlobalWall & keep, const GlobalWall & other) const
  {
    if (std::min(keep.hits, other.hits) < cfg_.merge_min_hits) {return false;}
    // Same face: each first seen from the side the other was, and the robot on that side
    if (other.observer_xy && !keep.same_side(other.observer_xy->x(), other.observer_xy->y())) {return false;}
    if (keep.observer_xy && !other.same_side(keep.observer_xy->x(), keep.observer_xy->y())) {return false;}
    if (!keep.same_side(robot_x_, robot_y_)) {return false;}
    const WallDiff ko = diff_calc(other.polar(), keep.polar());  // other's endpoints to keep's line
    const WallDiff ok = diff_calc(keep.polar(), other.polar());  // and the reverse
    return ko.angle < merge_angle_thresh_ && std::max(ko.dist, ok.dist) < cfg_.merge_dist_thresh &&
           ko.gap < cfg_.merge_gap_thresh;
  }

  GlobalWall * find_wall(int id)
  {
    for (auto & w : global_walls_) {
      if (w.id == id) {return &w;}
    }
    return nullptr;
  }

  // Merge each wall updated this frame with any stable wall it has grown into (older id kept)
  void merge_grown_walls(std::set<int> updated)
  {
    bool merged = true;
    while (merged) {
      merged = false;
      for (int wid : updated) {
        GlobalWall * wall = find_wall(wid);
        if (!wall) {continue;}
        for (auto & other : global_walls_) {
          if (other.id == wid) {continue;}
          GlobalWall & keep = wall->id < other.id ? *wall : other;
          GlobalWall & gone = wall->id < other.id ? other : *wall;
          if (!merge_candidate(keep, gone)) {continue;}
          const double lateral = diff_calc(gone.polar(), keep.polar()).dist;
          std::ostringstream seg;
          seg << "[" << std::lround(keep.endpoints[0]) << ", " << std::lround(keep.endpoints[1]) << ", "
              << std::lround(keep.endpoints[2]) << ", " << std::lround(keep.endpoints[3]) << "] + ["
              << std::lround(gone.endpoints[0]) << ", " << std::lround(gone.endpoints[1]) << ", "
              << std::lround(gone.endpoints[2]) << ", " << std::lround(gone.endpoints[3]) << "] cm";
          keep.absorb(gone);
          const int keep_id = keep.id, gone_id = gone.id;
          const double length = keep.d2 - keep.d1;
          global_walls_.erase(std::remove_if(global_walls_.begin(), global_walls_.end(),
            [gone_id](const GlobalWall & w) {return w.id == gone_id;}), global_walls_.end());
          // Walls bridged to the removed one are now bridged to the kept one
          auto & kept_bridges = merge_bridges_[keep_id];
          auto it = merge_bridges_.find(gone_id);
          if (it != merge_bridges_.end()) {
            kept_bridges.insert(kept_bridges.end(), it->second.begin(), it->second.end());
            merge_bridges_.erase(it);
          }
          kept_bridges.emplace_back(gone_id, cfg_.merge_bridge_obs);
          updated.erase(gone_id);
          updated.insert(keep_id);
          RCLCPP_INFO(get_logger(), "Merged wall ID %d into ID %d (lateral offset %.1f cm, length %.0f cm): %s",
            gone_id, keep_id, lateral, length, seg.str().c_str());
          merged = true;
          break;
        }
        if (merged) {break;}
      }
    }
  }

  // Duplicate observations of merged walls under the removed ids, with split information
  std::vector<WallObs> add_bridge_observations(const std::vector<WallObs> & obs)
  {
    std::vector<WallObs> out;
    for (const auto & o : obs) {
      auto it = merge_bridges_.find(o.id);
      std::vector<std::pair<int, int> *> live;
      if (it != merge_bridges_.end()) {
        for (auto & b : it->second) {
          if (b.second > 0) {live.push_back(&b);}
        }
      }
      if (live.empty()) {
        out.push_back(o);
        continue;
      }
      // n + 1 edges from one measurement: each with (n + 1) times the variance
      const double split = o.cov_scale * (live.size() + 1);
      out.push_back({o.id, o.rel_wall, split});
      for (auto * b : live) {
        out.push_back({b->first, o.rel_wall, split});
        b->second -= 1;
      }
      auto & v = it->second;
      v.erase(std::remove_if(v.begin(), v.end(), [](const auto & b) {return b.second <= 0;}), v.end());
    }
    return out;
  }

  // ---------------------------------------------------------------- callback
  void callback(const msg_interfaces::msg::MapElementswithDistance & msg)
  {
    const rclcpp::Time current_time(msg.header.stamp, get_clock()->get_clock_type());
    bool turning_too_fast = false;
    // Movement compensation using TF
    if (cfg_.movement_flag && last_msg_time_) {
      geometry_msgs::msg::TransformStamped trans;
      try {
        // Wait up to 0.1 s of wall time for the transform: the ROS clock (sim time) does not
        // advance while this callback blocks, so Buffer's own timeout could never expire.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (!tf_buffer_->canTransform(cfg_.map_frame, cfg_.lidar_frame, current_time) &&
          std::chrono::steady_clock::now() < deadline)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        trans = tf_buffer_->lookupTransform(cfg_.map_frame, cfg_.lidar_frame, current_time);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(get_logger(), "TF transform failed: %s", ex.what());
        return;
      }
      const double raw_x = trans.transform.translation.x * 100.0;  // m -> cm
      const double raw_y = trans.transform.translation.y * 100.0;
      const auto & q = trans.transform.rotation;
      const double raw_yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      if (is_first_frame_) {
        robot_x_ = raw_x; robot_y_ = raw_y; robot_yaw_ = raw_yaw;
        last_tf_x_ = raw_x; last_tf_y_ = raw_y; last_tf_yaw_ = raw_yaw;
        is_first_frame_ = false;
      } else {
        const double dx = raw_x - last_tf_x_;
        const double dy = raw_y - last_tf_y_;
        const double dyaw = wrap_pi(raw_yaw - last_tf_yaw_);
        last_tf_x_ = raw_x; last_tf_y_ = raw_y; last_tf_yaw_ = raw_yaw;
        robot_x_ += dx;
        robot_y_ += dy;
        travel_since_map_update_m_ += std::hypot(dx, dy) / 100.0;
        robot_yaw_ = pymod(robot_yaw_ + dyaw, kTwoPi);

        const double dt = (current_time - *last_msg_time_).seconds();
        if (cfg_.max_yaw_rate_deg_s > 0.0 && dt > 0.0 && std::abs(rad2deg(dyaw)) / dt > cfg_.max_yaw_rate_deg_s) {
          turning_too_fast = true;
          last_fast_turn_time_ = current_time;
        } else if (last_fast_turn_time_ &&
          (current_time - *last_fast_turn_time_).seconds() < cfg_.yaw_rate_settle_sec)
        {
          turning_too_fast = true;
        }
      }
    }
    robot_trajectory_.emplace_back(robot_x_, robot_y_);
    last_msg_time_ = current_time;
    if (turning_too_fast) {
      return;  // pose is tracked above; skip mapping and publishing for this frame
    }

    std::vector<Polar> local;
    for (size_t i = 0; i + 3 < msg.wall_parameters.size(); i += 4) {
      local.push_back({msg.wall_parameters[i], msg.wall_parameters[i + 1],
          msg.wall_parameters[i + 2], msg.wall_parameters[i + 3]});
    }
    std::vector<std::array<double, 3>> local_cols;
    for (size_t i = 0; i + 2 < msg.columns_xyr.size(); i += 3) {
      local_cols.push_back({msg.columns_xyr[i], msg.columns_xyr[i + 1], msg.columns_xyr[i + 2]});
    }
    local = manhattan_pre_filter(local);

    // Frame consistency: offset applied for this frame only
    const double saved_x = robot_x_, saved_y = robot_y_, saved_yaw = robot_yaw_;
    unverified_revisit_ = false;
    if (cfg_.frame_consistency_flag) {
      Eigen::Vector3d off;
      const OffsetResult r = estimate_frame_offset(local, off);
      if (r == OffsetResult::kUnverified) {
        unverified_revisit_ = true;
        suppress_revisit_slack_ = true;  // no consistent offset: base gates only
      } else if (r == OffsetResult::kOffset) {
        robot_x_ += off(0);
        robot_y_ += off(1);
        robot_yaw_ = pymod(robot_yaw_ + off(2), kTwoPi);
        suppress_revisit_slack_ = true;
        RCLCPP_DEBUG(get_logger(), "Frame offset dx %.1f dy %.1f cm, dyaw %.2f deg", off(0), off(1), rad2deg(off(2)));
      }
    }

    auto [matches, unmatched] = associate(local);

    // Map update
    const std::vector<Polar> final_walls = transform_walls(local);
    const auto final_cols = transform_columns(local_cols);

    std::set<int> matched_ids;
    std::vector<WallObs> wall_obs;
    for (const auto & [l, gid] : matches) {
      GlobalWall * target = find_wall(gid);
      if (!target) {continue;}
      target->update(final_walls[l][0], final_walls[l][1]);
      target->recalculate_endpoints_on_line(polar2endpoints(final_walls[l]));
      matched_ids.insert(target->id);
      if (target->hits >= 5) {
        wall_obs.push_back({target->id, local[l], 1.0});
      }
    }
    for (int l : unmatched) {
      const Polar & w = final_walls[l];
      global_walls_.emplace_back(next_wall_id_, w[0], w[1], w[2], w[3], Vec2(robot_x_, robot_y_));
      matched_ids.insert(next_wall_id_);
      ++next_wall_id_;
    }

    // Columns
    std::set<int> matched_col_ids;
    for (const auto & c : final_cols) {
      int best = -1;
      double best_dist = kInf;
      for (size_t j = 0; j < global_columns_.size(); ++j) {
        const double d = std::hypot(c[0] - global_columns_[j].x, c[1] - global_columns_[j].y);
        if (d < cfg_.match_dist_thresh && d < best_dist) {
          best_dist = d;
          best = static_cast<int>(j);
        }
      }
      if (best != -1) {
        global_columns_[best].update(c[0], c[1], c[2]);
        matched_col_ids.insert(global_columns_[best].id);
      } else {
        global_columns_.emplace_back(next_column_id_, c[0], c[1], c[2]);
        matched_col_ids.insert(next_column_id_);
        ++next_column_id_;
      }
    }

    // Lifecycle
    std::vector<GlobalWall> keep_walls;
    for (auto & g : global_walls_) {
      if (matched_ids.count(g.id)) {
        g.misses = 0;
        g.stale_m *= cfg_.revisit_slack_decay;
        keep_walls.push_back(g);
        continue;
      }
      g.stale_m += travel_since_map_update_m_;
      const double cx = g.rho * std::cos(g.theta), cy = g.rho * std::sin(g.theta);
      if (std::hypot(cx - robot_x_, cy - robot_y_) < cfg_.sensor_range &&
        !is_occluded(cx, cy, final_walls, final_cols))
      {
        g.misses += 1;
      }
      if (g.misses < cfg_.max_misses_thresh || g.hits >= 5) {
        keep_walls.push_back(g);
      } else {
        RCLCPP_INFO(get_logger(), "Removing wall ID %d.", g.id);
      }
    }
    global_walls_ = std::move(keep_walls);
    travel_since_map_update_m_ = 0.0;

    if (cfg_.merge_flag) {
      merge_grown_walls(matched_ids);
      wall_obs = add_bridge_observations(wall_obs);
    }

    std::vector<GlobalColumn> keep_cols;
    for (auto & c : global_columns_) {
      if (matched_col_ids.count(c.id)) {
        c.misses = 0;
        keep_cols.push_back(c);
      } else if (std::hypot(c.x - robot_x_, c.y - robot_y_) > cfg_.sensor_range) {
        c.misses = 0;
        keep_cols.push_back(c);
      } else {
        if (std::hypot(c.x - robot_x_, c.y - robot_y_) < cfg_.sensor_range &&
          !is_occluded(c.x, c.y, final_walls, final_cols))
        {
          c.misses += 1;
        }
        if (c.misses < cfg_.max_misses_thresh) {keep_cols.push_back(c);}
      }
    }
    global_columns_ = std::move(keep_cols);

    if (cfg_.manhattan_opt_flag) {
      optimizer_.apply_global_topology(global_walls_);
    }

    robot_x_ = saved_x;
    robot_y_ = saved_y;
    robot_yaw_ = saved_yaw;
    suppress_revisit_slack_ = false;
    unverified_revisit_ = false;

    publish_observations(msg.header, wall_obs);

    const auto now_mono = std::chrono::steady_clock::now();
    if (cfg_.publish_img_flag && image_pub_->get_subscription_count() > 0 &&
      (!last_render_time_ ||
      std::chrono::duration<double>(now_mono - *last_render_time_).count() >= cfg_.image_period_sec))
    {
      last_render_time_ = now_mono;
      publish_debug_image(final_walls);
    }
  }

  void publish_observations(const std_msgs::msg::Header & header, const std::vector<WallObs> & obs)
  {
    dps_slam_msgs::msg::DetectionWithIDArray out;
    out.header.stamp = header.stamp;
    out.header.frame_id = cfg_.lidar_frame;
    for (const auto & o : obs) {
      const double rho = o.rel_wall[0], theta = o.rel_wall[1];
      if (std::isnan(rho) || std::isnan(theta)) {continue;}
      dps_slam_msgs::msg::DetectionWithID det;
      det.id = std::to_string(o.id);
      det.label = "wall";
      det.geometry.type = dps_slam_msgs::msg::Geometry::LINE;
      auto & line = det.geometry.line;
      line.normal.x = std::cos(theta);
      line.normal.y = std::sin(theta);
      line.normal.z = 0.0;
      line.distance = rho / 100.0;
      const Segment e = polar2endpoints(o.rel_wall);
      // Covariance [theta, distance], growing with range
      const double range_m = std::hypot(e[0] + e[2], e[1] + e[3]) / 200.0;  // midpoint, cm -> m
      const double dist_std_m = (cfg_.obs_dist_std_cm + cfg_.obs_dist_std_per_m_cm * range_m) / 100.0;
      // cov_scale > 1: one measurement sent under several ids (merge bridges)
      const double s_dd = std::max(dist_std_m * dist_std_m, 1e-4) * o.cov_scale;
      const double a = deg2rad(cfg_.obs_angle_std_deg);
      const double s_tt = std::max(a * a, 1e-4) * o.cov_scale;
      line.covariance = {s_tt, 0.0, 0.0, s_dd};
      geometry_msgs::msg::Point p1, p2;
      p1.x = e[0] / 100.0; p1.y = e[1] / 100.0;
      p2.x = e[2] / 100.0; p2.y = e[3] / 100.0;
      line.boundary = {p1, p2};
      out.detections.push_back(det);
    }
    obs_pub_->publish(out);
  }

  void publish_debug_image(const std::vector<Polar> & incoming)
  {
    const int img_size = cfg_.processed_img_size;
    std::vector<double> xs = {robot_x_}, ys = {robot_y_};
    for (const auto & [tx, ty] : robot_trajectory_) {xs.push_back(tx); ys.push_back(ty);}
    for (const auto & g : global_walls_) {
      if (g.hits < 2) {continue;}
      xs.insert(xs.end(), {g.endpoints[0], g.endpoints[2]});
      ys.insert(ys.end(), {g.endpoints[1], g.endpoints[3]});
    }
    for (const auto & c : global_columns_) {
      if (c.hits < 2) {continue;}
      xs.insert(xs.end(), {c.x - c.radius, c.x + c.radius});
      ys.insert(ys.end(), {c.y - c.radius, c.y + c.radius});
    }
    const double padding = 300.0;
    const double min_x = *std::min_element(xs.begin(), xs.end()) - padding;
    const double max_x = *std::max_element(xs.begin(), xs.end()) + padding;
    const double min_y = *std::min_element(ys.begin(), ys.end()) - padding;
    const double max_y = *std::max_element(ys.begin(), ys.end()) + padding;
    const double span = std::max({max_x - min_x, max_y - min_y, 1000.0});
    const double scale = img_size / span;
    const double cxr = (min_x + max_x) / 2.0, cyr = (min_y + max_y) / 2.0, ci = img_size / 2.0;
    auto to_img = [&](double x, double y) {
        return cv::Point(static_cast<int>((x - cxr) * scale + ci), static_cast<int>(ci - (y - cyr) * scale));
      };

    cv::Mat canvas(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
    if (robot_trajectory_.size() > 1) {
      std::vector<cv::Point> traj;
      for (const auto & [tx, ty] : robot_trajectory_) {traj.push_back(to_img(tx, ty));}
      cv::polylines(canvas, traj, false, cv::Scalar(0, 165, 255), 2);
    }
    for (const auto & w : incoming) {
      const Segment e = polar2endpoints(w);
      cv::line(canvas, to_img(e[0], e[1]), to_img(e[2], e[3]), cv::Scalar(255, 0, 0), 5);
    }
    for (const auto & g : global_walls_) {
      if (g.hits < 2) {continue;}
      const cv::Point p1 = to_img(g.endpoints[0], g.endpoints[1]), p2 = to_img(g.endpoints[2], g.endpoints[3]);
      cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 3);
      cv::putText(canvas, "ID:" + std::to_string(g.id), cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2 - 5),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1);
    }
    for (const auto & c : global_columns_) {
      if (c.hits < 2) {continue;}
      const cv::Point p = to_img(c.x, c.y);
      cv::circle(canvas, p, static_cast<int>(c.radius * scale), cv::Scalar(0, 255, 0), 2);
      cv::putText(canvas, "C:" + std::to_string(c.id), p + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.4,
        cv::Scalar(0, 100, 0), 1);
    }
    cv::circle(canvas, to_img(robot_x_, robot_y_), std::max(3, static_cast<int>(20 * scale)), cv::Scalar(0, 0, 0), -1);
    for (const auto & c : active_debug_corners_) {
      const cv::Point p = to_img(c.x(), c.y());
      cv::drawMarker(canvas, p, cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 15, 2);
      cv::putText(canvas, "Corner", p + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 2);
    }

    sensor_msgs::msg::Image img;
    img.header.stamp = now();
    img.header.frame_id = "lidar_frame";
    img.height = img_size;
    img.width = img_size;
    img.encoding = "bgr8";
    img.step = img_size * 3;
    img.data.assign(canvas.data, canvas.data + canvas.total() * canvas.elemSize());
    image_pub_->publish(img);
  }

  GlobalFusionConfig cfg_;
  double match_angle_thresh_;
  double merge_angle_thresh_;
  ManhattanWorldOptimizer optimizer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  double robot_x_ = 0.0, robot_y_ = 0.0, robot_yaw_ = 0.0;
  std::optional<rclcpp::Time> last_msg_time_;
  std::optional<rclcpp::Time> last_fast_turn_time_;
  double travel_since_map_update_m_ = 0.0;  // m, for revisit gates
  bool suppress_revisit_slack_ = false;  // base gates only this frame
  bool unverified_revisit_ = false;  // revisit without consistent offset
  std::set<int> revisit_candidates_;
  double last_tf_x_ = 0.0, last_tf_y_ = 0.0, last_tf_yaw_ = 0.0;
  bool is_first_frame_ = true;

  int next_wall_id_ = 0;
  std::vector<GlobalWall> global_walls_;
  int next_column_id_ = 0;
  std::vector<GlobalColumn> global_columns_;
  // Merged walls: kept id -> [(removed id, observations still to bridge), ...]
  std::map<int, std::vector<std::pair<int, int>>> merge_bridges_;
  std::vector<Vec2> active_debug_corners_;
  std::vector<std::pair<double, double>> robot_trajectory_;
  std::optional<std::chrono::steady_clock::time_point> last_render_time_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<dps_slam_msgs::msg::DetectionWithIDArray>::SharedPtr obs_pub_;
  rclcpp::Subscription<msg_interfaces::msg::MapElementswithDistance>::SharedPtr sub_;
};

}  // namespace sslam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sslam::GlobalFusionNode>());
  rclcpp::shutdown();
  return 0;
}
