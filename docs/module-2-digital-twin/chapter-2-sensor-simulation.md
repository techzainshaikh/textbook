---
title: Sensor Simulation in Gazebo
sidebar_position: 3
description: Understanding sensor simulation for digital twins using Gazebo
keywords: [sensor simulation, gazebo, lidar, camera, imu, robotics simulation]
---

# Chapter 2: Sensor Simulation in Gazebo

## Learning Objectives

By the end of this chapter, students will be able to:
- Configure various sensor types in Gazebo simulation (LiDAR, cameras, IMU, GPS, etc.)
- Implement realistic sensor noise models to match real-world performance
- Validate sensor data quality and accuracy in simulation vs. real hardware
- Integrate simulated sensors with ROS 2 topics and message types
- Design custom sensor configurations for specific robotic applications

## Prerequisites

Students should have:
- Understanding of basic sensor types and their applications in robotics
- Knowledge of ROS 2 message types for sensor data (sensor_msgs package)
- Completed Chapter 1 (Physics Simulation in Gazebo)
- Basic understanding of probability and statistics for noise modeling

## Core Concepts

Sensor simulation is critical for developing robust perception and navigation systems. Realistic sensor simulation allows developers to test algorithms under various conditions before deploying to physical hardware.

### Types of Sensors in Robotics Simulation

**Range Sensors (LiDAR, Sonar, IR):**
- Provide distance measurements to obstacles
- Critical for navigation and mapping
- Subject to noise, occlusion, and range limitations

**Cameras (RGB, Depth, Stereo):**
- Provide visual information for object recognition and scene understanding
- Require significant computational resources
- Affected by lighting conditions and motion blur

**Inertial Measurement Units (IMU):**
- Measure acceleration and angular velocity
- Essential for localization and stabilization
- Drift over time, affected by bias and noise

**GPS and Positioning:**
- Provide global position estimates
- Accuracy varies with environmental conditions
- Affected by multipath and signal obstruction

### Noise Modeling

Real sensors are subject to various types of noise:
- **Gaussian noise**: Random variations around the true measurement
- **Bias**: Systematic offset in measurements
- **Drift**: Slow variation in sensor characteristics over time
- **Quantization**: Discretization effects in digital sensors

## Implementation

Let's implement sensor simulation for our humanoid robot. We'll configure realistic sensors with appropriate noise models.

### LiDAR Sensor Configuration

Here's an example of configuring a 3D LiDAR sensor in Gazebo:

```xml
<!-- LiDAR sensor configuration -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="humanoid_lidar">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>640</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle> <!-- -90 degrees -->
          <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.261799</min_angle> <!-- -15 degrees -->
          <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Sensor Configuration

Configuring a realistic RGB camera with noise parameters:

```xml
<!-- Camera sensor configuration -->
<gazebo reference="camera_link">
  <sensor type="camera" name="humanoid_camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <image_topic_name>image_raw</image_topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor Configuration

Configuring an IMU with realistic noise characteristics:

```xml
<!-- IMU sensor configuration -->
<gazebo reference="imu_link">
  <sensor type="imu" name="humanoid_imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev> <!-- ~0.1 deg/s -->
            <bias_mean>0.0001</bias_mean>
            <bias_stddev>0.00001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0001</bias_mean>
            <bias_stddev>0.00001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0001</bias_mean>
            <bias_stddev>0.00001</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev> <!-- ~0.0017 m/s² -->
            <bias_mean>0.01</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.01</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.01</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>torso</body_name>
    </plugin>
  </sensor>
</gazebo>
```

## Examples

### Example 1: Sensor Validation Node

Let's create a node to validate sensor data quality:

```python
#!/usr/bin/env python3
# sensor_validation.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from cv_bridge import CvBridge
import numpy as np
import cv2

class SensorValidator(Node):
    """Validate sensor data quality and accuracy"""

    def __init__(self):
        super().__init__('sensor_validator')

        # Subscribers for different sensor types
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.scan_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/humanoid/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu/data',
            self.imu_callback,
            10
        )

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Statistics tracking
        self.scan_stats = {'count': 0, 'range_errors': 0}
        self.image_stats = {'count': 0, 'quality_issues': 0}
        self.imu_stats = {'count': 0, 'drift_warnings': 0}

        # Timers for periodic reporting
        self.report_timer = self.create_timer(5.0, self.report_statistics)

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.scan_stats['count'] += 1

        # Check for invalid ranges
        invalid_ranges = 0
        for r in msg.ranges:
            if np.isnan(r) or np.isinf(r):
                invalid_ranges += 1

        if invalid_ranges > len(msg.ranges) * 0.1:  # More than 10% invalid
            self.get_logger().warn(f'High invalid range readings: {invalid_ranges}/{len(msg.ranges)}')
            self.scan_stats['range_errors'] += 1

        # Log scan quality metrics
        valid_ranges = [r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))]
        if valid_ranges:
            avg_range = sum(valid_ranges) / len(valid_ranges)
            self.get_logger().info(f'Avg scan range: {avg_range:.2f}m')

    def image_callback(self, msg):
        """Process camera image data"""
        self.image_stats['count'] += 1

        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Calculate image quality metrics
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Measure sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < 100:  # Low sharpness threshold
                self.get_logger().warn(f'Blurry image detected: sharpness={laplacian_var:.2f}')
                self.image_stats['quality_issues'] += 1

            # Measure brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 30 or mean_brightness > 220:  # Too dark or too bright
                self.get_logger().info(f'Image brightness: {mean_brightness:.2f}')

        except Exception as e:
            self.get_logger().error(f'Image processing error: {str(e)}')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_stats['count'] += 1

        # Check for unrealistic acceleration values (likely sensor error)
        lin_acc_mag = np.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )

        # Earth's gravity is ~9.81 m/s², so total acceleration shouldn't exceed ~15 m/s²
        # unless the robot is accelerating rapidly
        if lin_acc_mag > 15.0:
            self.get_logger().warn(f'High acceleration detected: {lin_acc_mag:.2f} m/s²')

        # Check angular velocity magnitude
        ang_vel_mag = np.sqrt(
            msg.angular_velocity.x**2 +
            msg.angular_velocity.y**2 +
            msg.angular_velocity.z**2
        )

        if ang_vel_mag > 10.0:  # Unusually high rotation rate
            self.get_logger().info(f'High angular velocity: {ang_vel_mag:.2f} rad/s')

    def report_statistics(self):
        """Report sensor validation statistics"""
        self.get_logger().info('=== Sensor Validation Report ===')
        self.get_logger().info(f'LiDAR: {self.scan_stats["count"]} scans, {self.scan_stats["range_errors"]} errors')
        self.get_logger().info(f'Camera: {self.image_stats["count"]} images, {self.image_stats["quality_issues"]} quality issues')
        self.get_logger().info(f'IMU: {self.imu_stats["count"]} samples, {self.imu_stats["drift_warnings"]} warnings')
        self.get_logger().info('===============================')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Multi-Sensor Fusion Node

```python
#!/usr/bin/env python3
# sensor_fusion.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from collections import deque

class SensorFusion(Node):
    """Fuse multiple sensor inputs for improved state estimation"""

    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for fused pose estimate
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/humanoid/pose_fused',
            10
        )

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # State estimation variables
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.acceleration_bias = np.zeros(3)  # Bias in acceleration measurements

        # Buffers for sensor fusion
        self.imu_buffer = deque(maxlen=10)
        self.last_scan_time = self.get_clock().now()

        # Timing
        self.prev_time = None

    def scan_callback(self, msg):
        """Process LiDAR scan for position correction"""
        # In a real implementation, we'd use scan matching or landmark detection
        # For this example, we'll just log the scan and use it to correct position estimates

        self.get_logger().info(f'Received scan with {len(msg.ranges)} beams')

        # Simple obstacle detection
        min_range = min([r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))], default=float('inf'))

        if min_range < 1.0:  # Obstacle within 1 meter
            self.get_logger().info(f'Obstacle detected at {min_range:.2f}m, adjusting position estimate')

    def imu_callback(self, msg):
        """Process IMU data for orientation and acceleration"""
        current_time = self.get_clock().now()

        if self.prev_time is not None:
            dt = (current_time - self.prev_time).nanoseconds / 1e9

            if dt > 0:
                # Extract acceleration (remove gravity)
                raw_acc = np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])

                # Apply bias correction
                corrected_acc = raw_acc - self.acceleration_bias

                # Update velocity using acceleration
                self.velocity += corrected_acc * dt

                # Update position using velocity
                self.position += self.velocity * dt + 0.5 * corrected_acc * dt**2

                # Extract orientation from quaternion
                self.orientation = np.array([
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ])

                # Publish fused pose estimate
                self.publish_pose_estimate(current_time)

        self.prev_time = current_time

    def publish_pose_estimate(self, timestamp):
        """Publish the fused pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp.to_msg()
        pose_msg.header.frame_id = 'odom'

        # Fill position
        pose_msg.pose.pose.position.x = self.position[0]
        pose_msg.pose.pose.position.y = self.position[1]
        pose_msg.pose.pose.position.z = self.position[2]

        # Fill orientation
        pose_msg.pose.pose.orientation.x = self.orientation[0]
        pose_msg.pose.pose.orientation.y = self.orientation[1]
        pose_msg.pose.pose.orientation.z = self.orientation[2]
        pose_msg.pose.pose.orientation.w = self.orientation[3]

        # Set covariance (diagonal values only for simplicity)
        cov_diag = [0.1, 0.1, 0.1, 0.01, 0.01, 0.1]  # px, py, pz, orx, ory, orz
        for i, val in enumerate(cov_diag):
            pose_msg.pose.covariance[i*7] = val  # Diagonal elements in 6x6 covariance matrix

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusion()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Sensor simulation is a critical component of digital twin technology in robotics. Properly configured sensors with realistic noise models allow for robust algorithm development and testing. Key considerations include:

- **Realistic Noise Models**: Matching simulation noise characteristics to real hardware
- **Update Rates**: Configuring appropriate sensor update rates for the application
- **Data Quality**: Validating sensor data for accuracy and consistency
- **Multi-Sensor Fusion**: Combining data from multiple sensors for improved state estimation
- **Calibration**: Ensuring simulated sensors are properly calibrated relative to the robot frame

## Exercises

### Conceptual
1. Explain the importance of realistic noise modeling in sensor simulation and how it affects algorithm development.

### Logical
1. Analyze the trade-offs between sensor accuracy and computational performance in simulation. How would you prioritize different sensor types for a humanoid robot navigation system?

### Implementation
1. Create a complete sensor simulation setup for a humanoid robot with LiDAR, camera, and IMU sensors, and implement a basic sensor validation node that monitors data quality.