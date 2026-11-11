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

#include "sslam_tools_cpp/assignment.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace sslam
{

std::vector<int> linear_sum_assignment(const Eigen::MatrixXd & cost)
{
  const int n = static_cast<int>(cost.rows());
  if (cost.cols() != n) {
    throw std::invalid_argument("linear_sum_assignment: cost matrix must be square");
  }
  if (n == 0) {
    return {};
  }
  // Forbidden pairs get a cost above any finite assignment, so they are never chosen
  // while a finite one exists.
  double finite_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::isfinite(cost(i, j))) {finite_sum += std::abs(cost(i, j));}
    }
  }
  const double big = 10.0 * (finite_sum + 1.0);
  auto c = [&](int i, int j) {
      const double v = cost(i, j);
      return std::isfinite(v) ? v : big;
    };

  // Shortest augmenting path with potentials (Hungarian, O(n^3)), 1-based internally.
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0), minv(n + 1);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);
  std::vector<bool> used(n + 1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::fill(minv.begin(), minv.end(), inf);
    std::fill(used.begin(), used.end(), false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) {continue;}
        const double cur = c(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> col(n, -1);
  for (int j = 1; j <= n; ++j) {
    col[p[j] - 1] = j - 1;
  }
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(cost(i, col[i]))) {
      throw std::runtime_error("linear_sum_assignment: cost matrix is infeasible");
    }
  }
  return col;
}

}  // namespace sslam
