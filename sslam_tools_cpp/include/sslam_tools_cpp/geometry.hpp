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

#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace sslam
{

using Vec2 = Eigen::Vector2d;
// Wall segment as endpoints [x1, y1, x2, y2]
using Segment = std::array<double, 4>;
// Wall in polar form [d (rho), alpha (theta), d1, d2]: the line is n . p = d with
// n = (cos alpha, sin alpha); d1, d2 are the endpoint positions along (-sin alpha, cos alpha).
using Polar = std::array<double, 4>;

constexpr double kPi = M_PI;
constexpr double kTwoPi = 2.0 * M_PI;

// Python's float modulo: the result has the sign of b (a % b in Python).
inline double pymod(double a, double b)
{
  double r = std::fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
    r += b;
  }
  return r;
}

inline double deg2rad(double deg) {return deg * kPi / 180.0;}
inline double rad2deg(double rad) {return rad * 180.0 / kPi;}
// Wrap to [-pi, pi)
inline double wrap_pi(double a) {return pymod(a + kPi, kTwoPi) - kPi;}

Polar endpoints2polar(const Segment & endpoints);
Segment polar2endpoints(const Polar & polar);

// Intersection of two lines in (d, alpha) form; nullopt when (nearly) parallel.
std::optional<Vec2> calculate_line_intersection(double d1, double alpha1, double d2, double alpha2);
// Same, solved as a 2x2 linear system (det threshold 1e-3).
std::optional<Vec2> intersection_cal_based_on_polar(double rho1, double theta1, double rho2, double theta2);

double point_to_segment_distance(double px, double py, double x1, double y1, double x2, double y2);
double distance_point_to_line(const Vec2 & point, const Vec2 & line_point1, const Vec2 & line_point2);
// Distance to the nearer endpoint of the segment (not the segment itself).
double distance_point_to_line_segment(const Vec2 & point, const Vec2 & line_point1, const Vec2 & line_point2);
// Along-wall gap between two segments, measured on w2's direction (0 when they overlap).
double overlap_cal(const Segment & w1, const Segment & w2);

double angle_logistic_function(double delta_angle, double angle_threshold, double k = 20.0, double m = 2.0);

// Skeletonization of the double-edge segments produced by LSD and Hough.
std::vector<Segment> merge_wall_envelopes(
  const std::vector<Segment> & segments, double thickness_thresh = 30.0,
  double angle_thresh_deg = 10.0, double overlap_thresh = 0.0);

// Wall centerline endpoints from the 4 vertices of an oriented box (midpoints of the short sides).
std::pair<Vec2, Vec2> calculate_walls_endpoints(const std::array<Vec2, 4> & vertices);
// Column center and radius from the 4 vertices of an oriented box.
std::pair<Vec2, double> calculate_columns_center_radius(const std::array<Vec2, 4> & vertices);

}  // namespace sslam
