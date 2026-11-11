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

#include <vector>

#include <Eigen/Core>

namespace sslam
{

// DBSCAN with the semantics of sklearn.cluster.DBSCAN: neighbours are within <= eps
// (the point itself included), a core point has >= min_samples neighbours, clusters are
// labelled 0, 1, ... in the order of their first core point, noise is -1.
std::vector<int> dbscan_precomputed(const Eigen::MatrixXd & dist, double eps, int min_samples);

// Same on 2D points with the euclidean metric (grid-accelerated neighbour search).
std::vector<int> dbscan_points(const std::vector<Eigen::Vector2d> & points, double eps, int min_samples);

}  // namespace sslam
