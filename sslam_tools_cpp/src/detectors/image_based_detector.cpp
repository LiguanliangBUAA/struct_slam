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

#include "sslam_tools_cpp/detectors/image_based_detector.hpp"

#include <opencv2/imgproc.hpp>

namespace sslam
{

void ImageBasedDetectorConfig::load(ParamReader & p)
{
  DetectorConfig::load(p);
  base_img_size = static_cast<int>(p.get_int("base_img_size", base_img_size));
  processed_img_size = static_cast<int>(p.get_int("processed_img_size", processed_img_size));
  sensor_posx = static_cast<int>(p.get_int("sensor_posx", sensor_posx));
  sensor_posy = static_cast<int>(p.get_int("sensor_posy", sensor_posy));
  kernel_size = static_cast<int>(p.get_int("kernel_size", kernel_size));
  iterations = static_cast<int>(p.get_int("iterations", iterations));
  smoothing = p.get_string("smoothing", smoothing);
  sigma = p.get_double("sigma", sigma);
  ksize_smoothing = static_cast<int>(p.get_int("ksize_smoothing", ksize_smoothing));
  last_dilation = p.get_string("last_dilation", last_dilation);
  truncate = p.get_bool("truncate", truncate);
  thresh = static_cast<int>(p.get_int("thresh", thresh));
}

ImageBasedDetector::ImageBasedDetector(std::string name, const ImageBasedDetectorConfig & cfg)
: Detector(std::move(name)), img_cfg_(cfg)
{
  ratio_bp_ = static_cast<double>(cfg.base_img_size) / cfg.processed_img_size;
  ratio_pr_ = static_cast<double>(cfg.result_img_size) / cfg.processed_img_size;
  raw_image_ = cv::Mat(cfg.processed_img_size, cfg.processed_img_size, CV_8UC1, cv::Scalar(255));
  processed_image_ = raw_image_.clone();
}

cv::Mat ImageBasedDetector::build_image(const std::vector<float> & x, const std::vector<float> & y,
  int img_size, int processed_img_size, int sensor_posx, int sensor_posy, int struct_radius)
{
  cv::Mat image(processed_img_size, processed_img_size, CV_8UC1, cv::Scalar(255));
  // float32 arithmetic and truncation toward zero, as numpy's astype(np.int32)
  const float scale = static_cast<float>(processed_img_size);
  const float size = static_cast<float>(img_size);
  for (size_t i = 0; i < x.size(); ++i) {
    const int px = static_cast<int>((static_cast<float>(sensor_posx) + x[i]) * scale / size);
    const int py = static_cast<int>((static_cast<float>(sensor_posy) - y[i]) * scale / size);
    if (px >= 0 && px < processed_img_size && py >= 0 && py < processed_img_size) {
      image.at<uint8_t>(py, px) = 0;
    }
  }
  const cv::Mat cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  if (struct_radius > 0) {
    cv::erode(image, image, cross, cv::Point(-1, -1), struct_radius);
  }
  return image;
}

void ImageBasedDetector::apply_smoothing(cv::Mat & image, const std::string & method, int ksize, double sigma)
{
  if (method == "none" || ksize <= 0) {
    return;
  }
  if (method == "gaussian") {
    cv::GaussianBlur(image, image, cv::Size(ksize, ksize), sigma);
  } else if (method == "median") {
    cv::medianBlur(image, image, ksize % 2 == 1 ? ksize : ksize + 1);
  } else if (method == "bilateral") {
    cv::Mat out;
    cv::bilateralFilter(image, out, ksize, sigma * 10, sigma * 10);
    image = out;
  } else if (method == "box") {
    cv::blur(image, image, cv::Size(ksize, ksize));
  }
}

void ImageBasedDetector::image_noise_processing(cv::Mat & image, const ImageBasedDetectorConfig & cfg)
{
  const cv::Mat kernel = cv::Mat::ones(cfg.kernel_size, cfg.kernel_size, CV_8U);
  apply_smoothing(image, cfg.smoothing, cfg.ksize_smoothing, cfg.sigma);
  if (cfg.truncate) {
    cv::threshold(image, image, cfg.thresh, 255, cv::THRESH_BINARY);
  }
  if (cfg.iterations > 0) {
    cv::erode(image, image, kernel, cv::Point(-1, -1), cfg.iterations);
  }
  apply_smoothing(image, cfg.last_dilation, cfg.ksize_smoothing, cfg.sigma);
}

SSCollection ImageBasedDetector::do_detect(const std::vector<float> & x, const std::vector<float> & y)
{
  raw_image_ = build_image(x, y, img_cfg_.base_img_size, img_cfg_.processed_img_size,
      img_cfg_.sensor_posx, img_cfg_.sensor_posy, img_cfg_.struct_radius);
  raw_image_.copyTo(processed_image_);
  image_noise_processing(processed_image_, img_cfg_);
  return inference(processed_image_);
}

Vec2 ImageBasedDetector::img_to_laser(double px, double py) const
{
  return Vec2(px * ratio_bp_ - img_cfg_.sensor_posx, img_cfg_.sensor_posy - py * ratio_bp_);
}

void ImageBasedDetector::processed_to_results_bgr()
{
  cv::Mat gray = processed_image_;
  if (processed_image_.rows != img_cfg_.result_img_size) {
    cv::resize(processed_image_, gray, cv::Size(img_cfg_.result_img_size, img_cfg_.result_img_size));
  }
  cv::cvtColor(gray, results_image_, cv::COLOR_GRAY2BGR);
}

}  // namespace sslam
