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


import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch_nodes(context):
    detectors_config = os.path.join(get_package_share_directory('sslam_tools'), 'config', 'detectors_config.yaml')
    lidar_config = os.path.join(get_package_share_directory('sslam_tools'), 'config', 'lidar_config.yaml')
    fusion_config = os.path.join(get_package_share_directory('sslam_tools'), 'config', 'fusion_config.yaml')
    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    # Loaded last, so its values override the package configs for whichever nodes it names.
    extra_params_file = LaunchConfiguration('extra_params_file').perform(context)
    extra = [extra_params_file] if extra_params_file else []

    return [
        Node(
            package='coord_conversion_cpp',
            executable='converter_livox',
            name='converter',
            parameters=[lidar_config, *extra, use_sim_time]
        ),
       Node(
           package='sslam_tools',
           executable='detector_node',
           parameters=[detectors_config, *extra, use_sim_time]
       ),
       Node(
           package='sslam_tools',
           executable='local_fusion',
           name='local_fusion_node',
           parameters=[fusion_config, *extra, use_sim_time]
       ),
       Node(
           package='sslam_tools',
           executable='global_fusion',
           name='global_fusion_node',
           parameters=[fusion_config, *extra, use_sim_time]
       ),
    ]

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', description='Use simulation clock', default_value='true')
    extra_params_file_arg = DeclareLaunchArgument(
        'extra_params_file', default_value='',
        description='Optional parameter file applied on top of the package configs '
                    '(e.g. overrides for real-flight data)')

    return LaunchDescription([
        use_sim_time_arg,
        extra_params_file_arg,
        OpaqueFunction(function=launch_nodes),
    ])
