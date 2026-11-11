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

#include "sslam_tools_cpp/global_map.hpp"

#include <algorithm>
#include <limits>

#include <Eigen/Dense>

namespace sslam
{

GlobalWall::GlobalWall(int id_, double rho_, double theta_, double d1_, double d2_,
  std::optional<Vec2> observer)
: id(id_), observer_xy(observer), rho(rho_), theta(theta_), d1(d1_), d2(d2_)
{
  endpoints = polar2endpoints({rho, theta, d1, d2});
  const double range_var = 10.0 * 10.0;  // cm^2
  const double angle_var = deg2rad(10.0) * deg2rad(10.0);  // rad^2
  covariance_state << range_var, 0, 0, angle_var;
  covariance_mea << range_var, 0, 0, angle_var;
}

void GlobalWall::update(double rho_mea, double theta_mea)
{
  // (rho, theta) and (-rho, theta + pi) are the same line. Express the measurement in the
  // state's form first, or a line near the origin can be blended ~90 deg off.
  if (std::cos(theta_mea - theta) < 0.0) {
    rho_mea = -rho_mea;
    theta_mea = theta_mea + kPi;
  }
  const Eigen::Matrix2d K = covariance_state * (covariance_state + covariance_mea).inverse();
  const Eigen::Vector2d diff(rho_mea - rho, wrap_pi(theta_mea - theta));
  const Eigen::Vector2d state = Eigen::Vector2d(rho, theta) + K * diff;
  rho = state(0);
  theta = pymod(state(1), kTwoPi);
  covariance_state = (Eigen::Matrix2d::Identity() - K) * covariance_state;
  hits += 1;
  misses = 0;
}

bool GlobalWall::same_side(double x, double y) const
{
  if (!observer_xy) {
    return true;
  }
  const Vec2 n(std::cos(theta), std::sin(theta));
  const double side_first = n.dot(*observer_xy) - rho;
  const double side_now = n.dot(Vec2(x, y)) - rho;
  return side_first * side_now > 0.0;
}

void GlobalWall::recalculate_endpoints_on_line(const Segment & raw)
{
  const Vec2 dir(-std::sin(theta), std::cos(theta));
  const Vec2 center(rho * std::cos(theta), rho * std::sin(theta));
  const std::array<Vec2, 4> pts = {Vec2(endpoints[0], endpoints[1]), Vec2(endpoints[2], endpoints[3]),
    Vec2(raw[0], raw[1]), Vec2(raw[2], raw[3])};
  double pmin = std::numeric_limits<double>::infinity();
  double pmax = -std::numeric_limits<double>::infinity();
  for (const auto & p : pts) {
    const double s = (p - center).dot(dir);
    pmin = std::min(pmin, s);
    pmax = std::max(pmax, s);
  }
  const double new_d1 = std::max(pmin, d1 - 50.0);
  const double new_d2 = std::min(pmax, d2 + 50.0);
  d1 = new_d1;
  d2 = new_d2;
  const Vec2 a = center + d1 * dir;
  const Vec2 b = center + d2 * dir;
  endpoints = {a.x(), a.y(), b.x(), b.y()};
}

void GlobalWall::absorb(const GlobalWall & other)
{
  const Vec2 dir(-std::sin(theta), std::cos(theta));
  const Vec2 center(rho * std::cos(theta), rho * std::sin(theta));
  const std::array<Vec2, 4> pts = {Vec2(endpoints[0], endpoints[1]), Vec2(endpoints[2], endpoints[3]),
    Vec2(other.endpoints[0], other.endpoints[1]), Vec2(other.endpoints[2], other.endpoints[3])};
  double pmin = std::numeric_limits<double>::infinity();
  double pmax = -std::numeric_limits<double>::infinity();
  for (const auto & p : pts) {
    const double s = (p - center).dot(dir);
    pmin = std::min(pmin, s);
    pmax = std::max(pmax, s);
  }
  d1 = pmin;
  d2 = pmax;
  const Vec2 a = center + d1 * dir;
  const Vec2 b = center + d2 * dir;
  endpoints = {a.x(), a.y(), b.x(), b.y()};
  hits += other.hits;
  misses = 0;
  stale_m = std::min(stale_m, other.stale_m);
}

void GlobalWall::recalculate_d_parameters()
{
  const Vec2 dir(-std::sin(theta), std::cos(theta));
  const Vec2 center(rho * std::cos(theta), rho * std::sin(theta));
  const Vec2 p_start(endpoints[0], endpoints[1]);
  const Vec2 p_end(endpoints[2], endpoints[3]);
  d1 = (p_start - center).dot(dir);
  d2 = (p_end - center).dot(dir);
  if (d1 > d2) {
    std::swap(d1, d2);
    endpoints = {p_end.x(), p_end.y(), p_start.x(), p_start.y()};
  }
}

GlobalColumn::GlobalColumn(int id_, double x_, double y_, double r_)
: id(id_), x(x_), y(y_), radius(r_) {}

void GlobalColumn::update(double x_mea, double y_mea, double radius_mea)
{
  x = (x * hits + x_mea) / (hits + 1);
  y = (y * hits + y_mea) / (hits + 1);
  radius = (radius * hits + radius_mea) / (hits + 1);
  hits += 1;
  misses = 0;
}

void ManhattanWorldOptimizer::apply_global_topology(std::vector<GlobalWall> & walls) const
{
  if (walls.empty()) {
    return;
  }
  apply_manhattan_constraints(walls, find_dominant_direction(walls));
  close_corners(walls);
}

double ManhattanWorldOptimizer::find_dominant_direction(const std::vector<GlobalWall> & walls) const
{
  if (walls.empty()) {
    return 0.0;
  }
  std::vector<double> angles, weights;
  for (const auto & w : walls) {
    angles.push_back(pymod(w.theta, kPi / 2));
    weights.push_back(w.hits * (w.d2 - w.d1));
  }
  constexpr int n_bins = 90;
  const double bin_width = (kPi / 2) / n_bins;
  std::vector<double> hist(n_bins, 0.0), smoothed(n_bins, 0.0);
  for (size_t k = 0; k < angles.size(); ++k) {
    hist[static_cast<int>(angles[k] / bin_width) % n_bins] += weights[k];  // angles in [0, pi/2)
  }
  for (int i = 0; i < n_bins; ++i) {
    smoothed[i] = 0.25 * hist[(i - 1 + n_bins) % n_bins] + 0.5 * hist[i] + 0.25 * hist[(i + 1) % n_bins];
  }
  const int peak = static_cast<int>(std::max_element(smoothed.begin(), smoothed.end()) - smoothed.begin());
  const double bin_center = (peak + 0.5) * bin_width;
  double rs = 0.0, rc = 0.0, rw = 0.0;
  for (size_t k = 0; k < angles.size(); ++k) {
    const double diff = pymod(angles[k] - bin_center + kPi / 4, kPi / 2) - kPi / 4;
    if (std::abs(diff) < deg2rad(5.0)) {
      rs += weights[k] * std::sin(4 * angles[k]);
      rc += weights[k] * std::cos(4 * angles[k]);
      rw += weights[k];
    }
  }
  if (rw > 0) {
    return pymod(std::atan2(rs, rc) / 4.0, kPi / 2);
  }
  return bin_center;
}

void ManhattanWorldOptimizer::apply_manhattan_constraints(
  std::vector<GlobalWall> & walls, double dominant_theta) const
{
  for (auto & wall : walls) {
    const double current = wall.theta;
    const double diff = pymod(current - dominant_theta, kTwoPi);
    const double k = std::nearbyint(diff / (kPi / 2));  // round half to even, as Python's round
    const double target = pymod(dominant_theta + k * (kPi / 2), kTwoPi);
    const double actual = std::abs(std::atan2(std::sin(current - target), std::cos(current - target)));
    if (actual < manhattan_theta_thresh_) {
      const double mid_x = (wall.endpoints[0] + wall.endpoints[2]) / 2.0;
      const double mid_y = (wall.endpoints[1] + wall.endpoints[3]) / 2.0;
      const double new_theta = pymod(target, kTwoPi);
      wall.theta = new_theta;
      wall.rho = mid_x * std::cos(new_theta) + mid_y * std::sin(new_theta);
      wall.recalculate_endpoints_on_line(wall.endpoints);
    }
  }
}

void ManhattanWorldOptimizer::close_corners(std::vector<GlobalWall> & walls) const
{
  const size_t n = walls.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      GlobalWall & w1 = walls[i];
      GlobalWall & w2 = walls[j];
      double angle_diff = std::abs(w1.theta - w2.theta);
      angle_diff = std::min(angle_diff, kTwoPi - angle_diff);
      if (angle_diff < min_intersection_angle_ || angle_diff > max_intersection_angle_) {
        continue;
      }
      const Vec2 s1(w1.endpoints[0], w1.endpoints[1]), e1(w1.endpoints[2], w1.endpoints[3]);
      const Vec2 s2(w2.endpoints[0], w2.endpoints[1]), e2(w2.endpoints[2], w2.endpoints[3]);
      // Pair types in Python's order: ss, se, es, ee (first minimum wins)
      const double dists[4] = {(s1 - s2).norm(), (s1 - e2).norm(), (e1 - s2).norm(), (e1 - e2).norm()};
      const int best = static_cast<int>(std::min_element(dists, dists + 4) - dists);
      if (dists[best] >= corner_connect_dist_thresh_) {
        continue;
      }
      const auto inter = intersection_cal_based_on_polar(w1.rho, w1.theta, w2.rho, w2.theta);
      if (!inter) {
        continue;
      }
      const bool w1_start = best < 2;
      const bool w2_start = best % 2 == 0;
      if (w1_start) {
        w1.endpoints[0] = inter->x(); w1.endpoints[1] = inter->y();
      } else {
        w1.endpoints[2] = inter->x(); w1.endpoints[3] = inter->y();
      }
      w1.recalculate_d_parameters();
      if (w2_start) {
        w2.endpoints[0] = inter->x(); w2.endpoints[1] = inter->y();
      } else {
        w2.endpoints[2] = inter->x(); w2.endpoints[3] = inter->y();
      }
      w2.recalculate_d_parameters();
    }
  }
}

}  // namespace sslam
