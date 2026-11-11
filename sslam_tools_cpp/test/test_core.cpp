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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "sslam_tools_cpp/assignment.hpp"
#include "sslam_tools_cpp/clustering.hpp"
#include "sslam_tools_cpp/geometry.hpp"
#include "sslam_tools_cpp/global_map.hpp"

using namespace sslam;

TEST(Geometry, PymodMatchesPython)
{
  EXPECT_DOUBLE_EQ(pymod(-1.0, kTwoPi), kTwoPi - 1.0);
  EXPECT_DOUBLE_EQ(pymod(7.0, kTwoPi), 7.0 - kTwoPi);
  EXPECT_DOUBLE_EQ(pymod(0.0, kPi), 0.0);
}

TEST(Geometry, PolarRoundTrip)
{
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-500.0, 500.0);
  for (int k = 0; k < 1000; ++k) {
    const Segment s = {u(rng), u(rng), u(rng), u(rng)};
    if (std::hypot(s[2] - s[0], s[3] - s[1]) < 1.0) {continue;}
    const Polar p = endpoints2polar(s);
    EXPECT_GE(p[0], 0.0);
    const Segment back = polar2endpoints(p);
    for (int i = 0; i < 4; ++i) {EXPECT_NEAR(back[i], s[i], 1e-9);}
  }
}

TEST(Geometry, KnownValues)
{
  // Horizontal wall y = 100 from x = -50 to 50: normal points to +y
  const Polar p = endpoints2polar({-50, 100, 50, 100});
  EXPECT_NEAR(p[0], 100.0, 1e-9);
  EXPECT_NEAR(p[1], kPi / 2, 1e-9);
  EXPECT_NEAR(p[2], 50.0, 1e-9);  // along (-sin a, cos a) = (-1, 0)
  EXPECT_NEAR(p[3], -50.0, 1e-9);
  EXPECT_NEAR(overlap_cal({0, 0, 100, 0}, {150, 0, 300, 0}), 50.0, 1e-9);
  EXPECT_NEAR(overlap_cal({0, 0, 100, 0}, {50, 0, 300, 0}), 0.0, 1e-9);
  const auto e = calculate_walls_endpoints({Vec2(0, 0), Vec2(0, 10), Vec2(100, 10), Vec2(100, 0)});
  EXPECT_NEAR(e.first.x(), 0.0, 1e-9);
  EXPECT_NEAR(e.first.y(), 5.0, 1e-9);
  EXPECT_NEAR(e.second.x(), 100.0, 1e-9);
  EXPECT_NEAR(e.second.y(), 5.0, 1e-9);
}

TEST(Assignment, MatchesBruteForce)
{
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(0.0, 10.0);
  std::bernoulli_distribution forbid(0.3);
  for (int trial = 0; trial < 200; ++trial) {
    const int n = 1 + trial % 6;
    Eigen::MatrixXd c(n, n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        c(i, j) = forbid(rng) ? std::numeric_limits<double>::infinity() : u(rng);
      }
    }
    // Brute force over permutations
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    do {
      double s = 0;
      for (int i = 0; i < n; ++i) {s += c(i, perm[i]);}
      best = std::min(best, s);
    } while (std::next_permutation(perm.begin(), perm.end()));
    if (!std::isfinite(best)) {
      EXPECT_THROW(linear_sum_assignment(c), std::runtime_error);
      continue;
    }
    const auto col = linear_sum_assignment(c);
    double s = 0;
    for (int i = 0; i < n; ++i) {s += c(i, col[i]);}
    EXPECT_NEAR(s, best, 1e-9);
  }
}

TEST(Clustering, PrecomputedLikeSklearn)
{
  // 0-1 close, 2 alone, 3-4 close; eps 0.5, min_samples 1
  Eigen::MatrixXd d(5, 5);
  d << 0, 0.3, 2, 2, 2,
    0.3, 0, 2, 2, 2,
    2, 2, 0, 2, 2,
    2, 2, 2, 0, 0.5,
    2, 2, 2, 0.5, 0;
  EXPECT_EQ(dbscan_precomputed(d, 0.5, 1), (std::vector<int>{0, 0, 1, 2, 2}));
  // min_samples 3: nobody is core
  EXPECT_EQ(dbscan_precomputed(d, 0.5, 3), (std::vector<int>{-1, -1, -1, -1, -1}));
}

TEST(Clustering, PointsTwoBlobs)
{
  std::vector<Eigen::Vector2d> pts;
  for (int i = 0; i < 10; ++i) {pts.emplace_back(i * 5.0, 0.0);}
  for (int i = 0; i < 10; ++i) {pts.emplace_back(1000.0 + i * 5.0, 0.0);}
  pts.emplace_back(500.0, 500.0);
  const auto labels = dbscan_points(pts, 10.0, 3);
  for (int i = 0; i < 10; ++i) {EXPECT_EQ(labels[i], 0);}
  for (int i = 10; i < 20; ++i) {EXPECT_EQ(labels[i], 1);}
  EXPECT_EQ(labels[20], -1);
}

TEST(GlobalMap, UpdateFlipsOppositeForm)
{
  GlobalWall w(0, 100.0, 0.0, -50.0, 50.0);
  // Same line expressed as (-rho, theta + pi)
  w.update(-100.0, kPi);
  EXPECT_NEAR(w.rho, 100.0, 1e-9);
  EXPECT_NEAR(wrap_pi(w.theta), 0.0, 1e-9);
  EXPECT_EQ(w.hits, 2);
}
