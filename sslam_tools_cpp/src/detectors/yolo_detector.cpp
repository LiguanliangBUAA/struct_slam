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

#include "sslam_tools_cpp/detectors/yolo_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace sslam
{

void YoloDetectorConfig::load(ParamReader & p)
{
  ImageBasedDetectorConfig::load(p);
  yolo_model_path = p.get_string("yolo_model_path", yolo_model_path);
  yolo_model_name = p.get_string("yolo_model_name", yolo_model_name);
  device = p.get_string("device", device);
  label = p.get_bool("label", label);
  conf_threshold = p.get_double("conf_threshold", conf_threshold);
  yolo_imgsz = static_cast<int>(p.get_int("yolo_imgsz", yolo_imgsz));
}

namespace
{

// ultralytics' per-class plotting colors (BGR)
cv::Scalar class_color(int cls)
{
  static const std::vector<cv::Scalar> palette = {
    {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255}, {49, 210, 207},
    {10, 249, 72}, {23, 204, 146}, {134, 219, 61}, {52, 147, 26}, {187, 212, 0}};
  return palette[static_cast<size_t>(cls) % palette.size()];
}

}  // namespace

YoloDetectorConfig YoloDetector::load_config(ParamReader & params)
{
  YoloDetectorConfig cfg;
  cfg.load(params);
  return cfg;
}

YoloDetector::YoloDetector(std::string name, ParamReader & params, const std::string & share_dir)
: YoloDetector(std::move(name), load_config(params), share_dir) {}

YoloDetector::YoloDetector(std::string name, const YoloDetectorConfig & cfg, const std::string & share_dir)
: ImageBasedDetector(std::move(name), cfg), cfg_(cfg)
{
  init_results_image();
  const fs::path dir = fs::path(cfg_.yolo_model_path).is_absolute() ?
    fs::path(cfg_.yolo_model_path) : fs::path(share_dir) / "resource" / "yolo_model";
  YoloBackendOptions opts;
  opts.model = (dir / cfg_.yolo_model_name).string();
  opts.device = cfg_.device;
  opts.conf_threshold = cfg_.conf_threshold;
  opts.imgsz = cfg_.yolo_imgsz;
  opts.fallback_imgsz = cfg_.processed_img_size;
  backend_ = make_yolo_backend(opts, backend_note_);
}

SSCollection YoloDetector::inference(const cv::Mat & processed_image)
{
  cv::cvtColor(processed_image, rgb_image_, cv::COLOR_GRAY2RGB);
  boxes_ = backend_->detect(rgb_image_);

  SSCollection ssc;
  const auto & names = backend_->names();
  for (const auto & box : boxes_) {
    const auto name = names.find(box.cls);
    if (name == names.end() || name->second != "walls") {
      continue;
    }
    std::array<Vec2, 4> laser;
    for (int i = 0; i < 4; ++i) {
      laser[i] = img_to_laser(box.vertices[i].x(), box.vertices[i].y());
    }
    const auto [p1, p2] = calculate_walls_endpoints(laser);
    ssc.endpoints.insert(ssc.endpoints.end(), {static_cast<float>(p1.x()), static_cast<float>(p1.y()),
        static_cast<float>(p2.x()), static_cast<float>(p2.y())});
  }
  return ssc;
}

void YoloDetector::build_debug_image()
{
  cv::Mat canvas;
  cv::cvtColor(rgb_image_, canvas, cv::COLOR_RGB2BGR);
  const int lw = std::max(static_cast<int>(std::round((canvas.rows + canvas.cols) / 2.0 * 0.003)), 2);
  const auto & names = backend_->names();
  for (const auto & b : boxes_) {
    std::vector<cv::Point> poly;
    for (const auto & v : b.vertices) {
      poly.emplace_back(static_cast<int>(std::lround(v.x())), static_cast<int>(std::lround(v.y())));
    }
    const cv::Scalar color = class_color(b.cls);
    cv::polylines(canvas, poly, true, color, lw);
    if (cfg_.label) {
      const auto name = names.find(b.cls);
      char text[64];
      std::snprintf(text, sizeof(text), "%s %.2f", name != names.end() ? name->second.c_str() : "?", b.conf);
      cv::putText(canvas, text, poly[0], cv::FONT_HERSHEY_SIMPLEX, lw / 3.0, color, std::max(lw - 1, 1));
    }
  }
  if (canvas.rows != cfg_.result_img_size || canvas.cols != cfg_.result_img_size) {
    cv::resize(canvas, results_image_, cv::Size(cfg_.result_img_size, cfg_.result_img_size));
  } else {
    canvas.copyTo(results_image_);
  }
}

}  // namespace sslam
