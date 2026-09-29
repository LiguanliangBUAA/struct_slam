# Copyright 2026 Universidad Politecnica de Madrid (UPM).
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

"""Republish odometry with a configured pose covariance.

dps_slam weighs each odometry edge by the covariance of the odometry message (with its
calculate_odom_covariance off). The odometry's own covariance does not say how much to trust
it against the wall observations, so this node sets it from parameters, next to the wall
observation covariance of global_fusion: e.g. a tiny attitude variance to keep FAST-LIO's
heading and let walls correct position only. A std <= 0 leaves that part as received.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry


class OdomCovarianceNode(Node):
    def __init__(self):
        super().__init__('odom_covariance_node')
        self.declare_parameter('input_topic', '/odom')
        self.declare_parameter('output_topic', '/odom_cov')
        self.declare_parameter('position_std_m', 0.0)      # x, y, z
        self.declare_parameter('attitude_std_deg', 0.0)    # roll, pitch, yaw
        self.position_var = self.get_parameter('position_std_m').value ** 2
        self.attitude_var = math.radians(self.get_parameter('attitude_std_deg').value) ** 2
        self.override_position = self.get_parameter('position_std_m').value > 0.0
        self.override_attitude = self.get_parameter('attitude_std_deg').value > 0.0

        output_topic = self.get_parameter('output_topic').value
        input_topic = self.get_parameter('input_topic').value
        self.publisher = self.create_publisher(Odometry, output_topic, qos_profile_sensor_data)
        self.create_subscription(Odometry, input_topic, self.callback, qos_profile_sensor_data)
        self.get_logger().info(
            f"'{input_topic}' -> '{output_topic}': position std "
            f"{'%.4f m' % math.sqrt(self.position_var) if self.override_position else 'as received'}, "
            f"attitude std "
            f"{'%.4f deg' % math.degrees(math.sqrt(self.attitude_var)) if self.override_attitude else 'as received'}")

    def callback(self, msg):
        cov = list(msg.pose.covariance)
        for i in range(6):
            for j in range(6):
                block_i, block_j = i < 3, j < 3
                overridden = (self.override_position if block_i else self.override_attitude) or \
                             (self.override_position if block_j else self.override_attitude)
                if overridden:
                    cov[6 * i + j] = 0.0
        if self.override_position:
            for i in range(3):
                cov[7 * i] = self.position_var
        if self.override_attitude:
            for i in range(3, 6):
                cov[7 * i] = self.attitude_var
        msg.pose.covariance = cov
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomCovarianceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
