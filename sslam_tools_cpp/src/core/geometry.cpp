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

#include "sslam_tools_cpp/geometry.hpp"

#include <algorithm>
#include <limits>

#include <Eigen/Dense>

namespace sslam
{

Polar endpoints2polar(const Segment & e)
{
  const double x1 = e[0], y1 = e[1], x2 = e[2], y2 = e[3];
  const double A = y1 - y2;
  const double B = x2 - x1;
  const double C = x1 * y2 - x2 * y1;
  const double denominator = std::hypot(A, B);

  if (denominator <= 1e-6) {  // points too close
    return {std::hypot(x1, y1), pymod(std::atan2(y1, x1), kTwoPi), 0.0, 0.0};
  }

  const double d = std::abs(C) / denominator;
  double nx, ny;
  if (C > 0) {
    nx = -A; ny = -B;
  } else {
    nx = A; ny = B;
  }
  const double alpha = pymod(std::atan2(ny, nx), kTwoPi);
  const double lx = -std::sin(alpha);
  const double ly = std::cos(alpha);
  return {d, alpha, x1 * lx + y1 * ly, x2 * lx + y2 * ly};
}

Segment polar2endpoints(const Polar & p)
{
  const double d = p[0], alpha = p[1], d1 = p[2], d2 = p[3];
  const double lx = -std::sin(alpha);
  const double ly = std::cos(alpha);
  const double nx = ly;
  const double ny = -lx;
  const double x0 = d * nx;
  const double y0 = d * ny;
  return {x0 + d1 * lx, y0 + d1 * ly, x0 + d2 * lx, y0 + d2 * ly};
}

std::optional<Vec2> calculate_line_intersection(double d1, double alpha1, double d2, double alpha2)
{
  const double det = std::sin(alpha2 - alpha1);
  if (std::abs(det) < 1e-3) {
    return std::nullopt;
  }
  const double x = (d1 * std::sin(alpha2) - d2 * std::sin(alpha1)) / det;
  const double y = (d2 * std::cos(alpha1) - d1 * std::cos(alpha2)) / det;
  return Vec2(x, y);
}

std::optional<Vec2> intersection_cal_based_on_polar(double rho1, double theta1, double rho2, double theta2)
{
  Eigen::Matrix2d A;
  A << std::cos(theta1), std::sin(theta1),
    std::cos(theta2), std::sin(theta2);
  if (std::abs(A.determinant()) < 1e-3) {
    return std::nullopt;
  }
  return Vec2(A.partialPivLu().solve(Eigen::Vector2d(rho1, rho2)));
}

double point_to_segment_distance(double px, double py, double x1, double y1, double x2, double y2)
{
  const double line_mag = std::hypot(x2 - x1, y2 - y1);
  if (line_mag < 1e-6) {
    return std::hypot(px - x1, py - y1);
  }
  const double u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag * line_mag);
  double ix, iy;
  if (u < 0) {
    ix = x1; iy = y1;
  } else if (u > 1) {
    ix = x2; iy = y2;
  } else {
    ix = x1 + u * (x2 - x1);
    iy = y1 + u * (y2 - y1);
  }
  return std::hypot(px - ix, py - iy);
}

double distance_point_to_line(const Vec2 & point, const Vec2 & a, const Vec2 & b)
{
  const Vec2 line_vec = b - a;
  const Vec2 point_vec = point - a;
  const double line_len = line_vec.norm();
  if (line_len < 1e-8) {
    return point_vec.norm();
  }
  const Vec2 unit = line_vec / line_len;
  const Vec2 nearest = a + point_vec.dot(unit) * unit;
  return (point - nearest).norm();
}

double distance_point_to_line_segment(const Vec2 & point, const Vec2 & a, const Vec2 & b)
{
  return std::min((point - a).norm(), (point - b).norm());
}

double overlap_cal(const Segment & w1, const Segment & w2)
{
  const double g_theta = endpoints2polar(w2)[1];
  const Vec2 dir(-std::sin(g_theta), std::cos(g_theta));
  const double p1 = Vec2(w1[0], w1[1]).dot(dir);
  const double p2 = Vec2(w1[2], w1[3]).dot(dir);
  const double q1 = Vec2(w2[0], w2[1]).dot(dir);
  const double q2 = Vec2(w2[2], w2[3]).dot(dir);
  const double overlap = std::min(std::max(p1, p2), std::max(q1, q2)) -
    std::max(std::min(p1, p2), std::min(q1, q2));
  return overlap > 0 ? 0.0 : std::abs(overlap);
}

double angle_logistic_function(double delta_angle, double angle_threshold, double k, double m)
{
  return m / (1.0 + std::exp(-k * (delta_angle - angle_threshold)));
}

namespace
{

struct LineData
{
  Vec2 p1, p2;
  double rho, theta, length;
};

bool check_longitudinal_overlap(const LineData & a, const LineData & b, double overlap_thresh)
{
  const Vec2 vec_a = a.p2 - a.p1;
  const double norm_a = vec_a.norm();
  if (norm_a < 1e-6) {
    return false;
  }
  const Vec2 dir_a = vec_a / norm_a;
  const double b1 = (b.p1 - a.p1).dot(dir_a);
  const double b2 = (b.p2 - a.p1).dot(dir_a);
  const double overlap = std::min(norm_a, std::max(b1, b2)) - std::max(0.0, std::min(b1, b2));
  return overlap > overlap_thresh;
}

}  // namespace

std::vector<Segment> merge_wall_envelopes(
  const std::vector<Segment> & segments, double thickness_thresh,
  double angle_thresh_deg, double overlap_thresh)
{
  if (segments.empty()) {
    return {};
  }
  const double angle_thresh = deg2rad(angle_thresh_deg);
  std::vector<bool> active(segments.size(), true);
  std::vector<LineData> lines;
  lines.reserve(segments.size());
  for (const auto & s : segments) {
    const Polar pol = endpoints2polar(s);
    lines.push_back({Vec2(s[0], s[1]), Vec2(s[2], s[3]), pol[0], pol[1],
        std::hypot(s[2] - s[0], s[3] - s[1])});
  }

  bool has_merged = true;
  while (has_merged) {
    has_merged = false;
    std::vector<size_t> indices;
    for (size_t i = 0; i < active.size(); ++i) {
      if (active[i]) {indices.push_back(i);}
    }
    const size_t n = indices.size();
    for (size_t i = 0; i < n; ++i) {
      const size_t idx_a = indices[i];
      if (!active[idx_a]) {continue;}
      const LineData & a = lines[idx_a];
      long best = -1;
      double min_dist = std::numeric_limits<double>::infinity();
      for (size_t j = i + 1; j < n; ++j) {
        const size_t idx_b = indices[j];
        if (!active[idx_b]) {continue;}
        const LineData & b = lines[idx_b];
        double delta_angle = std::abs(a.theta - b.theta);
        delta_angle = std::min(delta_angle, kTwoPi - delta_angle);
        if (delta_angle > angle_thresh) {continue;}
        const double dist = std::abs(distance_point_to_line((b.p1 + b.p2) / 2.0, a.p1, a.p2));
        if (dist > thickness_thresh) {continue;}
        if (!check_longitudinal_overlap(a, b, overlap_thresh)) {continue;}
        if (dist < min_dist) {
          min_dist = dist;
          best = static_cast<long>(idx_b);
        }
      }
      if (best != -1) {
        const LineData b = lines[best];
        const double total = a.length + b.length;
        const double new_angle = (a.theta * a.length + b.theta * b.length) / total;
        const Vec2 dir(-std::sin(new_angle), std::cos(new_angle));
        const std::array<Vec2, 4> pts = {a.p1, a.p2, b.p1, b.p2};
        const Vec2 centroid = (pts[0] + pts[1] + pts[2] + pts[3]) / 4.0;
        double smin = std::numeric_limits<double>::infinity();
        double smax = -std::numeric_limits<double>::infinity();
        for (const auto & p : pts) {
          const double s = (p - centroid).dot(dir);
          smin = std::min(smin, s);
          smax = std::max(smax, s);
        }
        const Vec2 new_p1 = centroid + smin * dir;
        const Vec2 new_p2 = centroid + smax * dir;
        const double normal = new_angle + kPi / 2;
        lines[idx_a] = {new_p1, new_p2,
          new_p1.x() * std::cos(normal) + new_p1.y() * std::sin(normal),
          pymod(new_angle, kTwoPi), (new_p2 - new_p1).norm()};
        active[best] = false;
        has_merged = true;
        break;
      }
    }
  }

  std::vector<Segment> result;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (active[i]) {
      const auto & l = lines[i];
      result.push_back({l.p1.x(), l.p1.y(), l.p2.x(), l.p2.y()});
    }
  }
  return result;
}

namespace
{

// Indices (1..3) of the vertices nearest and farthest from vertex 0, and their distances.
void nearest_farthest(const std::array<Vec2, 4> & v, int & min_idx, int & max_idx, double & max_val)
{
  double d[3];
  for (int i = 1; i < 4; ++i) {
    d[i - 1] = (v[i] - v[0]).norm();
  }
  min_idx = 0; max_idx = 0;
  double min_val = d[0];
  max_val = d[0];
  for (int i = 0; i < 3; ++i) {
    if (d[i] < min_val) {min_val = d[i]; min_idx = i;}
    if (d[i] > max_val) {max_val = d[i]; max_idx = i;}
  }
  min_idx += 1;
  max_idx += 1;
}

}  // namespace

std::pair<Vec2, Vec2> calculate_walls_endpoints(const std::array<Vec2, 4> & v)
{
  int min_idx, max_idx;
  double max_val;
  nearest_farthest(v, min_idx, max_idx, max_val);
  const int mid_idx = 6 - min_idx - max_idx;
  return {(v[0] + v[min_idx]) / 2.0, (v[mid_idx] + v[max_idx]) / 2.0};
}

std::pair<Vec2, double> calculate_columns_center_radius(const std::array<Vec2, 4> & v)
{
  int min_idx, max_idx;
  double max_val;
  nearest_farthest(v, min_idx, max_idx, max_val);
  return {(v[0] + v[max_idx]) / 2.0, max_val / 2.0};
}

}  // namespace sslam
