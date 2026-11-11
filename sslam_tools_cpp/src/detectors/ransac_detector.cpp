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

#include "sslam_tools_cpp/detectors/ransac_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "sslam_tools_cpp/clustering.hpp"
#include "sslam_tools_cpp/detectors/image_based_detector.hpp"

namespace sslam
{

void RANSACDetectorConfig::load(ParamReader & p)
{
  DetectorConfig::load(p);
  virtual_canvas_size = static_cast<int>(p.get_int("virtual_canvas_size", virtual_canvas_size));
  sensor_posx = static_cast<int>(p.get_int("sensor_posx", sensor_posx));
  sensor_posy = static_cast<int>(p.get_int("sensor_posy", sensor_posy));
  DBSCAN_eps = p.get_double("DBSCAN_eps", DBSCAN_eps);
  DBSCAN_min_samples = static_cast<int>(p.get_int("DBSCAN_min_samples", DBSCAN_min_samples));
  RANSAC_residual_threshold = p.get_double("RANSAC_residual_threshold", RANSAC_residual_threshold);
  RANSAC_min_samples = static_cast<int>(p.get_int("RANSAC_min_samples", RANSAC_min_samples));
  length_threshold = p.get_double("length_threshold", length_threshold);
}

namespace
{

// Least squares y = a x + b over the selected points (a = 0 when x has no spread).
void fit_line(const std::vector<Vec2> & pts, const std::vector<int> & idx, double & a, double & b)
{
  double mx = 0.0, my = 0.0;
  for (int i : idx) {mx += pts[i].x(); my += pts[i].y();}
  mx /= idx.size();
  my /= idx.size();
  double sxx = 0.0, sxy = 0.0;
  for (int i : idx) {
    sxx += (pts[i].x() - mx) * (pts[i].x() - mx);
    sxy += (pts[i].x() - mx) * (pts[i].y() - my);
  }
  a = sxx > 1e-12 ? sxy / sxx : 0.0;
  b = my - a * mx;
}

// R^2 of the line on the selected points (sklearn's LinearRegression.score)
double r2_score(const std::vector<Vec2> & pts, const std::vector<int> & idx, double a, double b)
{
  double my = 0.0;
  for (int i : idx) {my += pts[i].y();}
  my /= idx.size();
  double ss_res = 0.0, ss_tot = 0.0;
  for (int i : idx) {
    const double r = pts[i].y() - (a * pts[i].x() + b);
    ss_res += r * r;
    ss_tot += (pts[i].y() - my) * (pts[i].y() - my);
  }
  if (ss_tot <= 0.0) {
    return ss_res <= 0.0 ? 1.0 : 0.0;
  }
  return 1.0 - ss_res / ss_tot;
}

// sklearn's _dynamic_max_trials
double dynamic_max_trials(int n_inliers, int n_samples, int min_samples, double probability)
{
  const double inlier_ratio = static_cast<double>(n_inliers) / n_samples;
  const double nom = std::max(std::numeric_limits<double>::epsilon(), 1.0 - probability);
  const double denom = std::max(std::numeric_limits<double>::epsilon(),
      1.0 - std::pow(inlier_ratio, min_samples));
  if (nom == 1.0) {return 0.0;}
  if (denom == 1.0) {return std::numeric_limits<double>::infinity();}
  return std::abs(std::ceil(std::log(nom) / std::log(denom)));
}

}  // namespace

RansacLineFit ransac_line_fit(const std::vector<Vec2> & pts, double residual_threshold,
  int min_samples, std::mt19937 & rng, int max_trials)
{
  RansacLineFit out;
  const int n = static_cast<int>(pts.size());
  if (n < min_samples || min_samples < 1) {
    return out;
  }
  std::vector<int> all(n);
  std::iota(all.begin(), all.end(), 0);
  int best_n = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  std::vector<int> best_inliers;
  double needed = max_trials;
  for (int trial = 0; trial < max_trials && trial < needed; ++trial) {
    std::shuffle(all.begin(), all.end(), rng);
    const std::vector<int> subset(all.begin(), all.begin() + min_samples);
    double a, b;
    fit_line(pts, subset, a, b);
    std::vector<int> inliers;
    for (int i = 0; i < n; ++i) {
      if (std::abs(pts[i].y() - (a * pts[i].x() + b)) <= residual_threshold) {
        inliers.push_back(i);
      }
    }
    const int n_in = static_cast<int>(inliers.size());
    if (n_in == 0 || n_in < best_n) {
      continue;
    }
    const double score = r2_score(pts, inliers, a, b);
    if (n_in == best_n && score < best_score) {
      continue;
    }
    best_n = n_in;
    best_score = score;
    best_inliers = inliers;
    needed = dynamic_max_trials(best_n, n, min_samples, 0.99);
  }
  if (best_inliers.empty()) {
    return out;
  }
  fit_line(pts, best_inliers, out.slope, out.intercept);
  out.inliers.assign(n, false);
  for (int i : best_inliers) {out.inliers[i] = true;}
  out.ok = true;
  return out;
}

RANSACDetector::RANSACDetector(std::string name, ParamReader & params)
: Detector(std::move(name))
{
  cfg_.load(params);
  init_results_image();
}

SSCollection RANSACDetector::do_detect(const std::vector<float> & x, const std::vector<float> & y)
{
  x_ = x;
  y_ = y;
  SSCollection ssc;
  if (x.empty()) {
    last_ = ssc;
    return ssc;
  }
  std::vector<Vec2> pts(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    pts[i] = Vec2(x[i], y[i]);
  }
  const std::vector<int> labels = dbscan_points(pts, cfg_.DBSCAN_eps, cfg_.DBSCAN_min_samples);
  std::map<int, std::vector<Vec2>> clusters;
  for (size_t i = 0; i < pts.size(); ++i) {
    if (labels[i] != -1) {clusters[labels[i]].push_back(pts[i]);}
  }
  for (const auto & [label, cluster] : clusters) {
    (void)label;
    if (cluster.size() < 3) {continue;}
    const RansacLineFit fit = ransac_line_fit(cluster, cfg_.RANSAC_residual_threshold,
        cfg_.RANSAC_min_samples, rng_);
    if (!fit.ok) {continue;}
    std::vector<Vec2> inliers;
    for (size_t i = 0; i < cluster.size(); ++i) {
      if (fit.inliers[i]) {inliers.push_back(cluster[i]);}
    }
    if (inliers.size() < 2) {continue;}
    const Vec2 dir = Vec2(1.0, fit.slope).normalized();
    size_t imin = 0, imax = 0;
    double pmin = std::numeric_limits<double>::infinity();
    double pmax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < inliers.size(); ++i) {
      const double s = (inliers[i] - inliers[0]).dot(dir);
      if (s < pmin) {pmin = s; imin = i;}
      if (s > pmax) {pmax = s; imax = i;}
    }
    const Vec2 p1 = inliers[imin];
    const Vec2 p2 = inliers[imax];
    const double length = (p2 - p1).norm();
    if (length > cfg_.length_threshold) {
      ssc.endpoints.insert(ssc.endpoints.end(), {static_cast<float>(p1.x()), static_cast<float>(p1.y()),
          static_cast<float>(p2.x()), static_cast<float>(p2.y())});
    } else {
      Vec2 c = Vec2::Zero();
      for (const auto & p : inliers) {c += p;}
      c /= static_cast<double>(inliers.size());
      ssc.others_circles_centers.insert(ssc.others_circles_centers.end(),
        {static_cast<float>(c.x()), static_cast<float>(c.y())});
      ssc.others_circles_radius.push_back(static_cast<float>(length / 2));
    }
  }
  last_ = ssc;
  return ssc;
}

cv::Point RANSACDetector::laser_to_img(double x, double y) const
{
  const double ratio = static_cast<double>(cfg_.result_img_size) / cfg_.virtual_canvas_size;
  int px = static_cast<int>((cfg_.sensor_posx + x) * ratio);
  int py = static_cast<int>((cfg_.sensor_posy - y) * ratio);
  if (px < 0 || px >= cfg_.result_img_size) {px = -1;}
  if (py < 0 || py >= cfg_.result_img_size) {py = -1;}
  return {px, py};
}

void RANSACDetector::build_debug_image()
{
  const cv::Mat gray = ImageBasedDetector::build_image(x_, y_, cfg_.virtual_canvas_size,
      cfg_.result_img_size, cfg_.sensor_posx, cfg_.sensor_posy, cfg_.struct_radius);
  cv::cvtColor(gray, results_image_, cv::COLOR_GRAY2BGR);
  const auto & e = last_.endpoints;
  for (size_t i = 0; i + 3 < e.size(); i += 4) {
    cv::line(results_image_, laser_to_img(e[i], e[i + 1]), laser_to_img(e[i + 2], e[i + 3]),
      cv::Scalar(255, 0, 0), 2);
  }
  const double scale = static_cast<double>(cfg_.result_img_size) / cfg_.virtual_canvas_size;
  for (size_t i = 0; i < last_.others_circles_radius.size(); ++i) {
    cv::circle(results_image_,
      laser_to_img(last_.others_circles_centers[2 * i], last_.others_circles_centers[2 * i + 1]),
      static_cast<int>(last_.others_circles_radius[i] * scale), cv::Scalar(0, 255, 0), 2);
  }
}

}  // namespace sslam
