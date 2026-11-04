#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>

#include "elevation_grid_filter.hpp"

namespace structure_filter {

using elevation_grid_filter::GridHash;
using Cell = std::pair<int, int>;

// Removes small non-structural objects (people, poles, pipes, clutter): points are grouped into
// connected grid clusters, and clusters shorter than min_width along their principal axis are dropped.
// Input: points in the sensor frame, mm.
class StructureFilter {

public:
    StructureFilter(float grid_size, float min_width, int gap_cells)
        : grid_size_(grid_size), min_width_(min_width), gap_cells_(gap_cells) {}

    void filter(const std::vector<float>& in_x, const std::vector<float>& in_y, const std::vector<float>& in_z,
                std::vector<float>& out_x, std::vector<float>& out_y, std::vector<float>& out_z) {
        out_x.clear();
        out_y.clear();
        out_z.clear();

        if (in_x.empty() || in_x.size() != in_y.size() || in_x.size() != in_z.size()) {
            return; // Invalid input
        }

        // Occupied cells, -1 = not yet assigned to a cluster
        std::unordered_map<Cell, int, GridHash> label;
        for (size_t i = 0; i < in_x.size(); ++i) {
            label.emplace(cell_of(in_x[i], in_y[i]), -1);
        }

        // Connected clusters, kept if long enough
        std::unordered_map<Cell, bool, GridHash> keep;
        std::vector<Cell> stack, cluster;
        const int reach = 1 + gap_cells_;
        for (auto& [seed, seed_label] : label) {
            if (seed_label != -1) continue;
            seed_label = 0;
            stack.assign(1, seed);
            cluster.clear();
            while (!stack.empty()) {
                Cell c = stack.back();
                stack.pop_back();
                cluster.push_back(c);
                for (int dx = -reach; dx <= reach; ++dx) {
                    for (int dy = -reach; dy <= reach; ++dy) {
                        auto it = label.find(Cell(c.first + dx, c.second + dy));
                        if (it != label.end() && it->second == -1) {
                            it->second = 0;
                            stack.push_back(it->first);
                        }
                    }
                }
            }
            bool wide = principal_length(cluster) >= min_width_;
            for (const auto& c : cluster) keep[c] = wide;
        }

        for (size_t i = 0; i < in_x.size(); ++i) {
            if (keep[cell_of(in_x[i], in_y[i])]) {
                out_x.push_back(in_x[i]);
                out_y.push_back(in_y[i]);
                out_z.push_back(in_z[i]);
            }
        }
    }

private:
    Cell cell_of(float x, float y) const {
        return Cell(static_cast<int>(std::floor(x / grid_size_)), static_cast<int>(std::floor(y / grid_size_)));
    }

    // Length of the cluster along its principal axis, mm
    float principal_length(const std::vector<Cell>& cells) const {
        double mx = 0.0, my = 0.0;
        for (const auto& c : cells) { mx += c.first; my += c.second; }
        mx /= cells.size();
        my /= cells.size();
        double sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (const auto& c : cells) {
            double dx = c.first - mx, dy = c.second - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        double angle = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
        double ux = std::cos(angle), uy = std::sin(angle);
        double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
        for (const auto& c : cells) {
            double p = c.first * ux + c.second * uy;
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        return static_cast<float>((hi - lo + 1.0) * grid_size_);
    }

    float grid_size_;  // mm
    float min_width_;  // mm
    int gap_cells_;
};

}
