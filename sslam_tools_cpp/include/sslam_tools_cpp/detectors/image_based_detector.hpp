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

#include <opencv2/core.hpp>

#include "sslam_tools_cpp/detectors/detector.hpp"
#include "sslam_tools_cpp/geometry.hpp"

namespace sslam
{

struct ImageBasedDetectorConfig : DetectorConfig
{
  int base_img_size = 4096;
  int processed_img_size = 640;
  int sensor_posx = 2048;
  int sensor_posy = 2048;

  int kernel_size = 3;
  int iterations = 1;
  std::string smoothing = "none";  // none, gaussian, median, bilateral, box
  double sigma = 1.0;
  int ksize_smoothing = 1;
  std::string last_dilation = "none";  // same options as smoothing
  bool truncate = true;
  int thresh = 210;

  void load(ParamReader & p);
};

// Rasterizes the scan into a BEV image and runs an image detector on it.
class ImageBasedDetector : public Detector
{
public:
  ImageBasedDetector(std::string name, const ImageBasedDetectorConfig & cfg);

  // White image with a black (eroded) pixel per point, processed_img_size square.
  static cv::Mat build_image(const std::vector<float> & x, const std::vector<float> & y,
    int img_size, int processed_img_size, int sensor_posx, int sensor_posy, int struct_radius);
  static void apply_smoothing(cv::Mat & image, const std::string & method, int ksize, double sigma);
  static void image_noise_processing(cv::Mat & image, const ImageBasedDetectorConfig & cfg);

protected:
  SSCollection do_detect(const std::vector<float> & x, const std::vector<float> & y) override;
  virtual SSCollection inference(const cv::Mat & processed_image) = 0;
  // Processed-image pixel -> laser frame (cm)
  Vec2 img_to_laser(double px, double py) const;
  // processed_image_ (gray) resized to result_img_size, as BGR, into results_image_
  void processed_to_results_bgr();

  ImageBasedDetectorConfig img_cfg_;
  double ratio_bp_;  // base / processed
  double ratio_pr_;  // result / processed
  cv::Mat raw_image_;
  cv::Mat processed_image_;
};

}  // namespace sslam
