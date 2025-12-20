---
title: Environment Modeling for Digital Twins
sidebar_position: 4
description: Creating realistic environments for simulation and testing in digital twins
keywords: [environment modeling, gazebo worlds, unity environments, robotics simulation]
---

# Chapter 3: Environment Modeling for Digital Twins

## Learning Objectives

By the end of this chapter, students will be able to:
- Design and create realistic 3D environments for robotics simulation
- Implement complex world files with varied terrains, obstacles, and interactive elements
- Configure lighting, weather, and environmental conditions in simulation
- Integrate environmental sensors with physics properties for realistic interaction
- Validate environment models against real-world scenarios

## Prerequisites

Students should have:
- Understanding of 3D coordinate systems and transformations
- Basic knowledge of physics simulation (covered in Chapter 1)
- Experience with sensor simulation (covered in Chapter 2)
- Familiarity with XML/SDF format for Gazebo world files

## Core Concepts

Environment modeling is fundamental to digital twin technology, as it creates the virtual space where robots operate. A well-designed environment enables comprehensive testing of robot capabilities under various conditions.

### Key Components of Environment Modeling

**Terrain and Ground Surfaces:**
- Flat surfaces for basic navigation
- Complex terrain with elevation changes
- Varied friction coefficients for different surfaces
- Dynamic surfaces that can change during simulation

**Static Obstacles:**
- Walls, furniture, and architectural elements
- Proper collision properties for realistic interaction
- Visual appearance that matches the intended application

**Dynamic Elements:**
- Moving obstacles that simulate people or vehicles
- Interactive objects that robots can manipulate
- Environmental changes (doors opening/closing, lights changing)

**Environmental Conditions:**
- Lighting variations (day/night, indoor/outdoor)
- Weather effects (rain, fog, wind)
- Acoustic properties for sound-based sensors

### SDF World File Structure

The Simulation Description Format (SDF) is used to define Gazebo environments:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_environment">
    <!-- World properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Environment objects -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1.0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## Implementation

Let's implement a comprehensive environment model for our humanoid robot. We'll create a multi-room environment with various challenges and interactive elements.

### Basic Indoor Environment

First, let's create a basic indoor environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_hallway">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.6</mu>
                <mu2>0.6</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="wall_front">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 -2.5 1.25 0 0 0</pose>
    </model>

    <model name="wall_back">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 2.5 1.25 0 0 0</pose>
    </model>

    <model name="wall_left">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>-5 0 1.25 0 0 0</pose>
    </model>

    <model name="wall_right">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>5 0 1.25 0 0 0</pose>
    </model>

    <!-- Obstacles -->
    <model name="obstacle1">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.9 0.2 0.2 1</ambient>
            <diffuse>0.9 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>-2 0 0.5 0 0 0</pose>
    </model>

    <model name="obstacle2">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.9 1</ambient>
            <diffuse>0.2 0.2 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>2 1 0.25 0 0 0</pose>
    </model>

    <!-- Spawn point for robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1.0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Advanced Environment with Multiple Rooms

For more complex scenarios, we can create interconnected rooms:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_room_house">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Environment properties -->
    <scene>
      <ambient>0.3 0.3 0.3 1</ambient>
      <background>0.6 0.7 0.8 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.7 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.1 -0.9</direction>
    </light>

    <!-- Indoor lighting -->
    <light name="room_light1" type="point">
      <pose>-2 0 2 0 0 0</pose>
      <diffuse>0.9 0.9 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.5</constant>
        <linear>0.1</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>

    <light name="room_light2" type="point">
      <pose>2 0 2 0 0 0</pose>
      <diffuse>0.9 0.9 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.5</constant>
        <linear>0.1</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>

    <!-- Ground plane -->
    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 10</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.7 0.6 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Room walls -->
    <!-- Living room walls -->
    <model name="living_wall_front">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 -5 1.25 0 0 0</pose>
    </model>

    <model name="living_wall_back">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 0 1.25 0 0 0</pose>
    </model>

    <model name="living_wall_left">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>-4 -2.5 1.25 0 0 0</pose>
    </model>

    <model name="living_wall_right">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>4 -2.5 1.25 0 0 0</pose>
    </model>

    <!-- Kitchen walls -->
    <model name="kitchen_wall_front">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 5 1.25 0 0 0</pose>
    </model>

    <model name="kitchen_wall_back">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 0 1.25 0 0 0</pose>
    </model>

    <model name="kitchen_wall_left">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>-4 2.5 1.25 0 0 0</pose>
    </model>

    <model name="kitchen_wall_right">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 5 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>4 2.5 1.25 0 0 0</pose>
    </model>

    <!-- Doorway between rooms -->
    <model name="doorway">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.2 2.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.2 2.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.2 1</ambient>
            <diffuse>0.5 0.3 0.2 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>0 0 1.25 0 0 0</pose>
    </model>

    <!-- Furniture -->
    <model name="table">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.2 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.2 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>2 -3 0.4 0 0 0</pose>
    </model>

    <model name="chair">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.6 1</ambient>
            <diffuse>0.4 0.2 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      <pose>2.5 -3.5 0.4 0 0 0</pose>
    </model>

    <!-- Spawn point for robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 -3 1.0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## Examples

### Example 1: Environment Validation Node

Let's create a node to validate environment properties and test robot navigation:

```python
#!/usr/bin/env python3
# environment_validator.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer
import numpy as np
import math

class EnvironmentValidator(Node):
    """Validate environment properties and test robot navigation"""

    def __init__(self):
        super().__init__('environment_validator')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # Publisher for navigation commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/humanoid/cmd_vel', 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.navigation_targets = [
            (-2.0, -3.0, 0.0),  # Living room target
            (2.0, 3.0, 0.0),    # Kitchen target
            (0.0, 0.0, 0.0)     # Center of hallway
        ]
        self.current_target_idx = 0
        self.target_tolerance = 0.5

        # Timers
        self.nav_timer = self.create_timer(0.1, self.navigate_to_target)
        self.status_timer = self.create_timer(5.0, self.report_environment_status)

        # Statistics
        self.stats = {
            'distance_traveled': 0.0,
            'collisions_detected': 0,
            'obstacles_avoided': 0,
            'navigation_success': 0
        }

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        # Check for obstacles in front of robot
        front_scan_idx = len(msg.ranges) // 2
        front_range = msg.ranges[front_scan_idx]

        if front_range < 0.8:  # Obstacle within 80cm
            self.get_logger().warn(f'Obstacle detected in front: {front_range:.2f}m')
            self.stats['obstacles_avoided'] += 1

            # Trigger avoidance maneuver
            self.execute_avoidance_maneuver()

    def odom_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose.pose

    def navigate_to_target(self):
        """Navigate towards current target"""
        if not self.current_pose:
            return

        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        # Get current target
        target = self.navigation_targets[self.current_target_idx]
        target_x, target_y, target_z = target

        # Calculate distance to target
        dist_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        if dist_to_target < self.target_tolerance:
            self.get_logger().info(f'Reached target {self.current_target_idx}: ({target_x}, {target_y})')
            self.stats['navigation_success'] += 1
            self.current_target_idx = (self.current_target_idx + 1) % len(self.navigation_targets)
            return

        # Calculate desired velocity
        cmd_vel = Twist()

        # Proportional controller for navigation
        k_linear = 0.5
        k_angular = 1.0

        # Linear velocity proportional to distance
        cmd_vel.linear.x = min(k_linear * dist_to_target, 0.5)

        # Calculate angle to target
        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)

        # Current orientation (simplified - assuming z-axis rotation)
        current_yaw = 2 * math.asin(self.current_pose.orientation.z)

        # Angular error
        angle_error = angle_to_target - current_yaw
        # Normalize angle to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd_vel.angular.z = k_angular * angle_error

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def execute_avoidance_maneuver(self):
        """Execute obstacle avoidance maneuver"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0  # Stop forward motion
        cmd_vel.angular.z = 0.5  # Turn slightly
        self.cmd_vel_pub.publish(cmd_vel)

        # After short delay, resume navigation
        self.get_logger().info('Executing obstacle avoidance maneuver')

    def report_environment_status(self):
        """Report environment validation statistics"""
        self.get_logger().info('=== Environment Validation Report ===')
        self.get_logger().info(f'Distance traveled: {self.stats["distance_traveled"]:.2f}m')
        self.get_logger().info(f'Collisions detected: {self.stats["collisions_detected"]}')
        self.get_logger().info(f'Obstacles avoided: {self.stats["obstacles_avoided"]}')
        self.get_logger().info(f'Navigation successes: {self.stats["navigation_success"]}')
        self.get_logger().info('=====================================')

def main(args=None):
    rclpy.init(args=args)
    validator = EnvironmentValidator()

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

### Example 2: Environment Change Monitor

```python
#!/usr/bin/env python3
# environment_monitor.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import numpy as np
import cv2
from cv_bridge import CvBridge

class EnvironmentMonitor(Node):
    """Monitor environmental changes and detect anomalies"""

    def __init__(self):
        super().__init__('environment_monitor')

        # Subscribers
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

        # Publishers
        self.anomaly_pub = self.create_publisher(PointStamped, '/humanoid/environment/anomalies', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/humanoid/environment/visualization', 10)

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Environment baseline (will be established over time)
        self.baseline_scans = []
        self.baseline_images = []
        self.max_baseline_samples = 10

        # Anomaly detection parameters
        self.anomaly_threshold = 0.2  # Threshold for detecting environmental changes
        self.environment_map = {}     # Store known environment features

        # Timers
        self.update_timer = self.create_timer(1.0, self.update_environment_model)

    def scan_callback(self, msg):
        """Process laser scan and detect environmental changes"""
        # Store scan for baseline calculation
        if len(self.baseline_scans) < self.max_baseline_samples:
            self.baseline_scans.append(list(msg.ranges))
            self.get_logger().info(f'Collecting baseline scan {len(self.baseline_scans)}/{self.max_baseline_samples}')
            return

        # Calculate average baseline
        avg_baseline = np.mean(self.baseline_scans, axis=0)

        # Compare current scan with baseline
        current_scan = np.array(msg.ranges)

        # Mask invalid ranges
        valid_mask = ~(np.isnan(current_scan) | np.isinf(current_scan))
        baseline_valid = avg_baseline[valid_mask]
        current_valid = current_scan[valid_mask]

        if len(baseline_valid) > 0:
            # Calculate difference
            diff = np.abs(baseline_valid - current_valid)
            max_diff = np.max(diff)

            if max_diff > self.anomaly_threshold:
                self.get_logger().warn(f'ENVIRONMENT CHANGE DETECTED: max diff = {max_diff:.2f}m')

                # Find the index of maximum difference
                max_idx = np.argmax(diff)

                # Calculate the angle for that range measurement
                angle_increment = (msg.angle_max - msg.angle_min) / len(msg.ranges)
                angle = msg.angle_min + max_idx * angle_increment

                # Convert to Cartesian coordinates relative to robot
                range_val = current_valid[max_idx]
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                # Publish anomaly location
                anomaly_point = PointStamped()
                anomaly_point.header.stamp = self.get_clock().now().to_msg()
                anomaly_point.header.frame_id = 'base_link'
                anomaly_point.point.x = x
                anomaly_point.point.y = y
                anomaly_point.point.z = 0.0

                self.anomaly_pub.publish(anomaly_point)

                # Visualize the anomaly
                self.visualize_anomaly(x, y)

    def image_callback(self, msg):
        """Process camera image and detect visual changes"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Store image for baseline if needed
            if len(self.baseline_images) < self.max_baseline_samples:
                self.baseline_images.append(cv_image.copy())
                return

            # Compare with baseline images using optical flow
            baseline_img = self.baseline_images[-1]  # Use most recent baseline

            # Convert to grayscale
            gray_current = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_baseline = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)

            # Calculate difference
            diff_img = cv2.absdiff(gray_current, gray_baseline)

            # Calculate average difference
            avg_diff = np.mean(diff_img)

            if avg_diff > 10:  # Threshold for significant visual change
                self.get_logger().info(f'VISUAL CHANGE DETECTED: avg diff = {avg_diff:.2f}')

        except Exception as e:
            self.get_logger().error(f'Image processing error: {str(e)}')

    def visualize_anomaly(self, x, y):
        """Visualize detected anomaly in RViz"""
        marker_array = MarkerArray()

        # Create a marker for the anomaly location
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'environment_anomalies'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5  # At robot eye level
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2  # 20cm sphere
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = 1.0  # Red color for anomaly
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Semi-transparent

        marker.lifetime.sec = 5  # Visible for 5 seconds

        marker_array.markers.append(marker)

        # Publish the visualization
        self.viz_pub.publish(marker_array)

    def update_environment_model(self):
        """Update internal environment model"""
        # This would typically update a map or other environment representation
        # For this example, we'll just log the current state

        if len(self.baseline_scans) >= self.max_baseline_samples:
            avg_scan = np.mean(self.baseline_scans, axis=0)
            self.get_logger().info(f'Environment model updated. Avg nearest obstacle: {np.min(avg_scan[np.isfinite(avg_scan)])}m')

def main(args=None):
    rclpy.init(args=args)
    monitor = EnvironmentMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Environment modeling is a critical aspect of digital twin technology for robotics. Well-designed environments enable comprehensive testing of robot capabilities under various conditions. Key considerations include:

- **Realistic Physics**: Ensuring proper friction, collision properties, and material characteristics
- **Visual Fidelity**: Balancing visual realism with computational performance
- **Interactive Elements**: Including objects and elements that robots can interact with
- **Environmental Variations**: Modeling different lighting, weather, and acoustic conditions
- **Scalability**: Designing environments that can be extended or modified as needed

## Exercises

### Conceptual
1. Explain the relationship between environment complexity and computational performance in robotics simulation.

### Logical
1. Analyze the trade-offs between environmental realism and simulation speed. When would you prioritize one over the other?

### Implementation
1. Create a complex multi-room environment with varied terrain types, obstacles, and interactive elements, and implement an environment monitoring node that detects changes in the environment.