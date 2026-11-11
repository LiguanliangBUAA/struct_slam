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

#include <opencv2/imgproc.hpp>

#include "sslam_tools_cpp/detectors/image_based_detector.hpp"

namespace sslam
{

class LSDDetector : public ImageBasedDetector
{
public:
  LSDDetector(std::string name, ParamReader & params);
  const DetectorConfig & config() const override {return img_cfg_;}

protected:
  SSCollection inference(const cv::Mat & processed_image) override;
  void build_debug_image() override;

private:
  static ImageBasedDetectorConfig load_config(ParamReader & params);

  cv::Ptr<cv::LineSegmentDetector> lsd_;
  std::vector<cv::Vec4f> lines_;
};

}  // namespace sslam
