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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace sslam
{

// Source of configuration values (the ROS node declares them as parameters). Each call
// returns the configured value, or the default when none is set.
class ParamReader
{
public:
  virtual ~ParamReader() = default;
  virtual bool get_bool(const std::string & name, bool def) = 0;
  virtual int64_t get_int(const std::string & name, int64_t def) = 0;
  virtual double get_double(const std::string & name, double def) = 0;
  virtual std::string get_string(const std::string & name, const std::string & def) = 0;
};

// Structural elements detected in one scan (laser frame, cm)
struct SSCollection
{
  std::vector<float> endpoints;  // [x1, y1, x2, y2] per wall
  std::vector<float> columns_circles_centers;  // [x, y] per column
  std::vector<float> columns_circles_radius;
  std::vector<float> others_circles_centers;
  std::vector<float> others_circles_radius;
};

struct DetectorConfig
{
  bool capture_inference_image = true;
  int result_img_size = 640;
  int struct_radius = 1;

  void load(ParamReader & p);
};

class Detector
{
public:
  explicit Detector(std::string name);
  virtual ~Detector() = default;

  // x, y: point coordinates in the laser frame (cm)
  SSCollection detect(const std::vector<float> & x, const std::vector<float> & y);

  const std::string & name() const {return name_;}
  virtual const DetectorConfig & config() const = 0;
  // Debug image of the last detection (BGR, result_img_size x result_img_size)
  const cv::Mat & results_image() const {return results_image_;}

protected:
  virtual SSCollection do_detect(const std::vector<float> & x, const std::vector<float> & y) = 0;
  virtual void build_debug_image() = 0;
  // Allocate the white debug image once the config is known
  void init_results_image();

  std::string name_;
  cv::Mat results_image_;
};

// Detector types: "yolo" (needs OpenVINO at build time), "lsd", "hought", "ransac".
// share_dir: installed package share directory (default location of the YOLO models).
std::unique_ptr<Detector> make_detector(
  const std::string & type, ParamReader & params, const std::string & share_dir);

}  // namespace sslam
