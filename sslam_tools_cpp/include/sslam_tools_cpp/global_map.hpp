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

#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "sslam_tools_cpp/geometry.hpp"

namespace sslam
{

class GlobalWall
{
public:
  GlobalWall(int id, double rho, double theta, double d1, double d2,
    std::optional<Vec2> observer_xy = std::nullopt);

  // Kalman update of (rho, theta) with a measurement (no prediction step).
  void update(double rho_mea, double theta_mea);
  // Whether (x, y) is on the side the wall was first observed from.
  bool same_side(double x, double y) const;
  // Re-project the extent on the updated line, growing by at most 50 cm per side.
  void recalculate_endpoints_on_line(const Segment & new_raw_endpoints);
  // Take over another segment of the same wall: extend the extent to cover both.
  // The line is kept, not averaged: the lateral offset is drift the SLAM backend resolves.
  void absorb(const GlobalWall & other);
  // Recompute d1, d2 from the endpoints (ordered so d1 <= d2).
  void recalculate_d_parameters();

  Polar polar() const {return {rho, theta, d1, d2};}

  int id;
  std::optional<Vec2> observer_xy;  // position the wall was first seen from
  double stale_m = 0.0;  // m flown since last matched (revisit gates)
  double rho, theta, d1, d2;
  Segment endpoints;
  int hits = 1;
  int misses = 0;
  Eigen::Matrix2d covariance_state;
  Eigen::Matrix2d covariance_mea;
};

class GlobalColumn
{
public:
  GlobalColumn(int id, double x, double y, double radius);
  void update(double x_mea, double y_mea, double radius_mea);

  int id;
  double x, y, radius;
  int hits = 1;
  int misses = 0;
};

class ManhattanWorldOptimizer
{
public:
  void apply_global_topology(std::vector<GlobalWall> & walls) const;
  double find_dominant_direction(const std::vector<GlobalWall> & walls) const;
  void apply_manhattan_constraints(std::vector<GlobalWall> & walls, double dominant_theta) const;
  void close_corners(std::vector<GlobalWall> & walls) const;

private:
  double manhattan_theta_thresh_ = deg2rad(5.0);  // wall adsorption angle threshold
  double corner_connect_dist_thresh_ = 10.0;  // corner closure threshold (cm)
  double min_intersection_angle_ = deg2rad(85.0);
  double max_intersection_angle_ = deg2rad(95.0);
};

}  // namespace sslam
