# Copyright 2026 Universidad Politecnica de Madrid (UPM).
#
# Author: Pedro Espinosa Angulo
# Contributor: Guanliang Li, Santiago Tapia Fernandez (supervised)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os, cv2
import numpy as np
import sslam_tools.geometry_functions as geo_funcs
from ament_index_python.packages import get_package_share_directory

from sslam_tools.detectors.detector import SSCollection
from sslam_tools.detectors.imagebased_detector import ImageBasedDetector, ImageBasedDetectorConfig
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class YoloDetectorConfig(ImageBasedDetectorConfig):
    # Relative paths are resolved against the package's installed share dir
    # (resource/yolo_model); an absolute path overrides that.
    yolo_model_path: str = ''
    yolo_model_name: str = 'YOLOv8n-obb_TRAINED_lgl_1_openvino_model'

    label: bool = True

class YoloDetector(ImageBasedDetector[YoloDetectorConfig]):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        # Resolve the model directory so it works regardless of the launch CWD.
        # Absolute config paths are used as-is; otherwise fall back to the
        # package's installed resource/yolo_model directory.
        if os.path.isabs(self.config.yolo_model_path):
            model_dir = self.config.yolo_model_path
        else:
            model_dir = os.path.join(
                get_package_share_directory('sslam_tools'), 'resource', 'yolo_model')

        file_path = os.path.join(model_dir, self.config.yolo_model_name)
        self.yoloModel: YOLO = YOLO(file_path, task='obb')
        self.rgb_image: np.ndarray = np.empty((self.config.processed_img_size, self.config.processed_img_size, 3), dtype=np.uint8)
        self.yolo_results: list = []

    @classmethod
    def get_config_class(cls) -> type[ImageBasedDetectorConfig]:
        return YoloDetectorConfig
    
    def inference(self, processed_image: np.ndarray) -> SSCollection:
        cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB, dst=self.rgb_image)
        self.yolo_results = self.yoloModel(self.rgb_image, device='cpu')

        ss_collection = SSCollection()
        for result in self.yolo_results:
            obb = result.obb
            for i in range(len(obb)):
                conf = float(obb.conf[i].cpu().numpy())
                if conf < 0.2: # Filter low-confidence detections
                    continue
                class_id = int(obb.cls[i].cpu().numpy())
                class_name = result.names[class_id]

                vertices_img = obb.xyxyxyxy[i].cpu().numpy()
                vertices_laser = self.img_to_laser(vertices_img)
                vertices = vertices_laser.flatten().tolist()

                if class_name in ['walls', 'shadows']:
                    vertices = np.array(vertices, dtype=np.float32).reshape((4, 2))
                    p1, p2 = geo_funcs.calculate_walls_endpoints(vertices)
                    # Flatten endpoints: [x1, y1, x2, y2]
                    end_points = [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]
                    ss_collection.endpoints.extend(end_points)
                elif class_name == 'columns':
                    vertices = np.array(vertices, dtype=np.float32).reshape((4, 2))
                    center, radius = geo_funcs.calculate_columns_center_radius(vertices)
                    center = center.flatten().tolist()
                    ss_collection.columns_circles_centers.extend(center)
                    ss_collection.columns_circles_radius.append(float(radius))
                elif class_name == 'others':
                    vertices = np.array(vertices, dtype=np.float32).reshape((4, 2))
                    center, radius = geo_funcs.calculate_columns_center_radius(vertices)
                    center = center.flatten().tolist()
                    ss_collection.others_circles_centers.extend(center)
                    ss_collection.others_circles_radius.append(float(radius))

        return ss_collection
    
    def _build_debug_image(self) -> np.ndarray:
        tmp = self.yolo_results[0].plot(labels=self.config.label, boxes=True)
        if self.use_tmp_rimage:
            cv2.resize(tmp, (self.config.result_img_size, self.config.result_img_size), dst=self.results_image)
        else:
            self.results_image = tmp
        return self.results_image