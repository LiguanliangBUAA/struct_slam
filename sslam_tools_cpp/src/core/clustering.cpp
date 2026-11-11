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

#include "sslam_tools_cpp/clustering.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace sslam
{

namespace
{

// sklearn's dbscan_inner: depth-first expansion from each unlabelled core point, in index order.
std::vector<int> expand(const std::vector<std::vector<int>> & neighborhoods, int min_samples)
{
  const int n = static_cast<int>(neighborhoods.size());
  std::vector<int> labels(n, -1);
  std::vector<bool> core(n);
  for (int i = 0; i < n; ++i) {
    core[i] = static_cast<int>(neighborhoods[i].size()) >= min_samples;
  }
  int label = 0;
  std::vector<int> stack;
  for (int i = 0; i < n; ++i) {
    if (labels[i] != -1 || !core[i]) {
      continue;
    }
    int p = i;
    while (true) {
      if (labels[p] == -1) {
        labels[p] = label;
        if (core[p]) {
          for (int q : neighborhoods[p]) {
            if (labels[q] == -1) {
              stack.push_back(q);
            }
          }
        }
      }
      if (stack.empty()) {
        break;
      }
      p = stack.back();
      stack.pop_back();
    }
    ++label;
  }
  return labels;
}

}  // namespace

std::vector<int> dbscan_precomputed(const Eigen::MatrixXd & dist, double eps, int min_samples)
{
  const int n = static_cast<int>(dist.rows());
  std::vector<std::vector<int>> neighborhoods(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (dist(i, j) <= eps) {
        neighborhoods[i].push_back(j);
      }
    }
  }
  return expand(neighborhoods, min_samples);
}

std::vector<int> dbscan_points(const std::vector<Eigen::Vector2d> & points, double eps, int min_samples)
{
  const int n = static_cast<int>(points.size());
  if (n == 0) {
    return {};
  }
  const double cell = eps > 0.0 ? eps : 1.0;
  auto key = [](int64_t cx, int64_t cy) {return (cx << 32) ^ (cy & 0xffffffff);};
  std::unordered_map<int64_t, std::vector<int>> grid;
  std::vector<int64_t> cx(n), cy(n);
  for (int i = 0; i < n; ++i) {
    cx[i] = static_cast<int64_t>(std::floor(points[i].x() / cell));
    cy[i] = static_cast<int64_t>(std::floor(points[i].y() / cell));
    grid[key(cx[i], cy[i])].push_back(i);
  }
  const double eps2 = eps * eps;
  std::vector<std::vector<int>> neighborhoods(n);
  for (int i = 0; i < n; ++i) {
    for (int64_t dx = -1; dx <= 1; ++dx) {
      for (int64_t dy = -1; dy <= 1; ++dy) {
        auto it = grid.find(key(cx[i] + dx, cy[i] + dy));
        if (it == grid.end()) {
          continue;
        }
        for (int j : it->second) {
          if ((points[i] - points[j]).squaredNorm() <= eps2) {
            neighborhoods[i].push_back(j);
          }
        }
      }
    }
  }
  return expand(neighborhoods, min_samples);
}

}  // namespace sslam
