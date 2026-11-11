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

#include "sslam_tools_cpp/detectors/detector.hpp"

#include <stdexcept>

#include "sslam_tools_cpp/detectors/hough_detector.hpp"
#include "sslam_tools_cpp/detectors/lsd_detector.hpp"
#include "sslam_tools_cpp/detectors/ransac_detector.hpp"
#include "sslam_tools_cpp/detectors/yolo_detector.hpp"

namespace sslam
{

void DetectorConfig::load(ParamReader & p)
{
  capture_inference_image = p.get_bool("capture_inference_image", capture_inference_image);
  result_img_size = static_cast<int>(p.get_int("result_img_size", result_img_size));
  struct_radius = static_cast<int>(p.get_int("struct_radius", struct_radius));
}

Detector::Detector(std::string name)
: name_(std::move(name)) {}

void Detector::init_results_image()
{
  const int s = config().result_img_size;
  results_image_ = cv::Mat(s, s, CV_8UC3, cv::Scalar(255, 255, 255));
}

SSCollection Detector::detect(const std::vector<float> & x, const std::vector<float> & y)
{
  if (x.size() != y.size()) {
    throw std::invalid_argument("x and y must have the same size");
  }
  SSCollection ssc = do_detect(x, y);
  if (config().capture_inference_image) {
    build_debug_image();
  }
  return ssc;
}

std::unique_ptr<Detector> make_detector(
  const std::string & type, ParamReader & params, const std::string & share_dir)
{
  const std::string name = type + "_detector";
  if (type == "yolo") {
    return std::make_unique<YoloDetector>(name, params, share_dir);
  }
  if (type == "lsd") {
    return std::make_unique<LSDDetector>(name, params);
  }
  if (type == "hought") {
    return std::make_unique<HoughTDetector>(name, params);
  }
  if (type == "ransac") {
    return std::make_unique<RANSACDetector>(name, params);
  }
  throw std::invalid_argument("Unsupported detector type: " + type);
}

}  // namespace sslam
