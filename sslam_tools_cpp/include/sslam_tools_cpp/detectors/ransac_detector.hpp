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

#include <random>
#include <string>
#include <vector>

#include "sslam_tools_cpp/detectors/detector.hpp"
#include "sslam_tools_cpp/geometry.hpp"

namespace sslam
{

struct RANSACDetectorConfig : DetectorConfig
{
  // Debug image
  int virtual_canvas_size = 4096;
  int sensor_posx = 2048;
  int sensor_posy = 2048;
  // Detector
  double DBSCAN_eps = 30.0;  // cm
  int DBSCAN_min_samples = 5;
  double RANSAC_residual_threshold = 15.0;  // cm
  int RANSAC_min_samples = 2;
  double length_threshold = 70.0;  // cm, walls vs other objects

  void load(ParamReader & p);
};

// Robust fit of y = slope * x + intercept (sklearn RANSACRegressor with a LinearRegression
// estimator, absolute-error residuals, max_trials 100, stop_probability 0.99).
struct RansacLineFit
{
  bool ok = false;
  double slope = 0.0;
  double intercept = 0.0;
  std::vector<bool> inliers;
};
RansacLineFit ransac_line_fit(const std::vector<Vec2> & pts, double residual_threshold,
  int min_samples, std::mt19937 & rng, int max_trials = 100);

// Clusters the points with DBSCAN and fits a line per cluster with RANSAC:
// long fits are walls, short ones "others" (circles).
class RANSACDetector : public Detector
{
public:
  RANSACDetector(std::string name, ParamReader & params);
  const DetectorConfig & config() const override {return cfg_;}

protected:
  SSCollection do_detect(const std::vector<float> & x, const std::vector<float> & y) override;
  void build_debug_image() override;

private:
  cv::Point laser_to_img(double x, double y) const;

  RANSACDetectorConfig cfg_;
  std::mt19937 rng_{42};
  std::vector<float> x_, y_;
  SSCollection last_;
};

}  // namespace sslam
