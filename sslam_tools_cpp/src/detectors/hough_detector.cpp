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

#include "sslam_tools_cpp/detectors/hough_detector.hpp"

#include <opencv2/imgproc.hpp>

namespace sslam
{

void HoughTDetectorConfig::load(ParamReader & p)
{
  ImageBasedDetectorConfig::load(p);
  rho = p.get_double("rho", rho);
  theta = p.get_double("theta", theta);
  threshold = static_cast<int>(p.get_int("threshold", threshold));
  min_line_length = p.get_double("min_line_length", min_line_length);
  max_line_gap = p.get_double("max_line_gap", max_line_gap);
}

HoughTDetectorConfig HoughTDetector::load_config(ParamReader & params)
{
  HoughTDetectorConfig cfg;
  cfg.load(params);
  return cfg;
}

HoughTDetector::HoughTDetector(std::string name, ParamReader & params)
: HoughTDetector(std::move(name), load_config(params), 0) {}

HoughTDetector::HoughTDetector(std::string name, const HoughTDetectorConfig & cfg, int)
: ImageBasedDetector(std::move(name), cfg), cfg_(cfg)
{
  init_results_image();
}

SSCollection HoughTDetector::inference(const cv::Mat & processed_image)
{
  cv::Canny(processed_image, edges_, 50, 200);
  lines_.clear();
  cv::HoughLinesP(edges_, lines_, cfg_.rho, cfg_.theta, cfg_.threshold,
    cfg_.min_line_length, cfg_.max_line_gap);
  SSCollection ssc;
  for (const auto & l : lines_) {
    const Vec2 p1 = img_to_laser(l[0], l[1]);
    const Vec2 p2 = img_to_laser(l[2], l[3]);
    ssc.endpoints.insert(ssc.endpoints.end(), {static_cast<float>(p1.x()), static_cast<float>(p1.y()),
        static_cast<float>(p2.x()), static_cast<float>(p2.y())});
  }
  return ssc;
}

void HoughTDetector::build_debug_image()
{
  processed_to_results_bgr();
  for (const auto & l : lines_) {
    cv::line(results_image_,
      cv::Point(static_cast<int>(l[0] * ratio_pr_), static_cast<int>(l[1] * ratio_pr_)),
      cv::Point(static_cast<int>(l[2] * ratio_pr_), static_cast<int>(l[3] * ratio_pr_)),
      cv::Scalar(0, 0, 255), 2);
  }
}

}  // namespace sslam
