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

#include "sslam_tools_cpp/detectors/lsd_detector.hpp"

namespace sslam
{

ImageBasedDetectorConfig LSDDetector::load_config(ParamReader & params)
{
  ImageBasedDetectorConfig cfg;
  cfg.load(params);
  return cfg;
}

LSDDetector::LSDDetector(std::string name, ParamReader & params)
: ImageBasedDetector(std::move(name), load_config(params))
{
  lsd_ = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
  init_results_image();
}

SSCollection LSDDetector::inference(const cv::Mat & processed_image)
{
  lines_.clear();
  lsd_->detect(processed_image, lines_);
  SSCollection ssc;
  for (const auto & l : lines_) {
    const Vec2 p1 = img_to_laser(l[0], l[1]);
    const Vec2 p2 = img_to_laser(l[2], l[3]);
    ssc.endpoints.insert(ssc.endpoints.end(), {static_cast<float>(p1.x()), static_cast<float>(p1.y()),
        static_cast<float>(p2.x()), static_cast<float>(p2.y())});
  }
  return ssc;
}

void LSDDetector::build_debug_image()
{
  processed_to_results_bgr();
  for (const auto & l : lines_) {
    cv::line(results_image_,
      cv::Point(static_cast<int>(l[0] * ratio_pr_), static_cast<int>(l[1] * ratio_pr_)),
      cv::Point(static_cast<int>(l[2] * ratio_pr_), static_cast<int>(l[3] * ratio_pr_)),
      cv::Scalar(0, 255, 0), 2);
  }
}

}  // namespace sslam
