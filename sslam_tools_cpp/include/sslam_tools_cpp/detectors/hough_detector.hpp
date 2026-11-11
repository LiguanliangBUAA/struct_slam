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

#include <string>
#include <vector>

#include "sslam_tools_cpp/detectors/image_based_detector.hpp"

namespace sslam
{

struct HoughTDetectorConfig : ImageBasedDetectorConfig
{
  double rho = 1.0;
  double theta = kPi / 180.0;
  int threshold = 50;
  double min_line_length = 10.0;
  double max_line_gap = 20.0;

  void load(ParamReader & p);
};

class HoughTDetector : public ImageBasedDetector
{
public:
  HoughTDetector(std::string name, ParamReader & params);
  const DetectorConfig & config() const override {return cfg_;}

protected:
  SSCollection inference(const cv::Mat & processed_image) override;
  void build_debug_image() override;

private:
  // Delegated to, so the parameters are read once (the int only disambiguates)
  HoughTDetector(std::string name, const HoughTDetectorConfig & cfg, int);
  static HoughTDetectorConfig load_config(ParamReader & params);

  HoughTDetectorConfig cfg_;
  cv::Mat edges_;
  std::vector<cv::Vec4i> lines_;
};

}  // namespace sslam
