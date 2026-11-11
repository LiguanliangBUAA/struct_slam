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

// Minimum-cost perfect assignment on a square cost matrix (scipy's linear_sum_assignment).
// Infinite entries are forbidden pairs; the matrix must admit a finite assignment.
// Returns col[i], the column assigned to row i.
std::vector<int> linear_sum_assignment(const Eigen::MatrixXd & cost);

}  // namespace sslam
