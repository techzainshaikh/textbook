---
title: Navigation and Path Planning with Isaac ROS
sidebar_position: 5
description: Implementing navigation and path planning systems using Isaac ROS and Nav2
keywords: [navigation, path planning, nav2, robotics, Isaac ROS, localization, mapping]
---

# Chapter 4: Navigation and Path Planning with Isaac ROS

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement navigation systems using Isaac ROS and Nav2 for humanoid robots
- Configure localization, mapping, and path planning algorithms
- Design costmap layers and obstacle avoidance strategies for humanoid navigation
- Integrate perception data with navigation for robust path planning
- Validate navigation performance in complex environments

## Prerequisites

Students should have:
- Understanding of mobile robot navigation concepts (localization, mapping, path planning)
- Knowledge of perception systems (covered in Chapter 3)
- Experience with ROS 2 navigation stack (Nav2)
- Basic understanding of control theory for motion execution

## Core Concepts

Navigation systems enable humanoid robots to autonomously move through environments safely and efficiently. Isaac ROS provides GPU-accelerated navigation capabilities that integrate with perception systems for robust autonomous navigation.

### Navigation System Architecture

**Localization Layer:**
- AMCL (Adaptive Monte Carlo Localization) for pose estimation
- Visual-inertial odometry for position tracking
- Sensor fusion for robust pose estimation
- Global and local map management

**Mapping Layer:**
- Occupancy grid mapping from sensor data
- Costmap construction for obstacle representation
- Dynamic obstacle tracking and prediction
- Map updates and maintenance

**Path Planning Layer:**
- Global planner for long-term path computation
- Local planner for obstacle avoidance and trajectory execution
- Recovery behaviors for navigation failures
- Dynamic path replanning for changing environments

**Motion Execution Layer:**
- Velocity controllers for smooth motion execution
- Footstep planning for bipedal humanoid navigation
- Balance control during motion execution
- Integration with robot dynamics

### Isaac ROS Navigation Framework

Isaac ROS provides specialized navigation capabilities:
- **Isaac ROS Nav2**: GPU-accelerated navigation algorithms
- **Isaac ROS Occupancy Grids**: High-resolution mapping
- **Isaac ROS Path Planners**: Optimized planning algorithms
- **Isaac ROS Controllers**: Specialized motion controllers for humanoid robots

## Implementation

Let's implement a comprehensive navigation system using Isaac ROS and Nav2:

### Isaac ROS Navigation Setup

```python
#!/usr/bin/env python3
# navigation_system.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
from scipy.spatial import distance
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class IsaacNavigationSystem(Node):
    """
    Navigation system using Isaac ROS and Nav2 for humanoid robot
    """

    def __init__(self):
        super().__init__('isaac_navigation_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/humanoid/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/humanoid/goal_pose', 10)
        self.path_pub = self.create_publisher(Path, '/humanoid/planned_path', 10)
        self.status_pub = self.create_publisher(String, '/humanoid/navigation/status', 10)
        self.debug_markers_pub = self.create_publisher(MarkerArray, '/humanoid/navigation/debug_markers', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/lidar/scan',
            self.scan_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/humanoid/map',
            self.map_callback,
            10
        )

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.current_goal = None
        self.global_path = []
        self.local_plan = []
        self.obstacles = []
        self.map_data = None

        # Navigation parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.goal_tolerance = 0.5  # meters
        self.obstacle_threshold = 0.7  # meters for obstacle detection
        self.replan_distance = 2.0  # distance at which to replan

        # Timers
        self.navigation_timer = self.create_timer(0.1, self.navigation_loop)  # 10 Hz
        self.path_planner_timer = self.create_timer(1.0, self.plan_path)  # Plan path periodically

        # Navigation modes
        self.navigation_active = False
        self.avoiding_obstacles = False

        self.get_logger().info('Isaac Navigation System initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Convert scan to obstacle points in robot frame
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid_indices = np.isfinite(ranges) & (ranges < msg.range_max) & (ranges > msg.range_min)
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]

        # Convert to Cartesian coordinates
        obstacle_x = valid_ranges * np.cos(valid_angles)
        obstacle_y = valid_ranges * np.sin(valid_angles)

        # Store obstacles
        self.obstacles = list(zip(obstacle_x, obstacle_y))

    def map_callback(self, msg):
        """Update internal map representation"""
        self.map_data = {
            'info': msg.info,
            'data': np.array(msg.data).reshape(msg.info.height, msg.info.width),
            'resolution': msg.info.resolution,
            'origin': (msg.info.origin.position.x, msg.info.origin.position.y)
        }

    def set_goal(self, x, y, theta=0.0):
        """Set navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0

        # Convert angle to quaternion
        goal_msg.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.orientation.w = math.cos(theta / 2.0)

        self.current_goal = goal_msg.pose
        self.navigation_active = True
        self.get_logger().info(f'Set navigation goal to ({x:.2f}, {y:.2f})')

    def navigation_loop(self):
        """Main navigation control loop"""
        if not self.current_pose or not self.current_goal or not self.navigation_active:
            return

        try:
            # Calculate distance to goal
            dx = self.current_goal.position.x - self.current_pose.position.x
            dy = self.current_goal.position.y - self.current_pose.position.y
            dist_to_goal = math.sqrt(dx*dx + dy*dy)

            # Check if goal reached
            if dist_to_goal < self.goal_tolerance:
                self.navigation_active = False
                self.get_logger().info('Goal reached!')

                # Stop robot
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)
                return

            # Check for obstacles in path
            obstacle_in_path = self.check_obstacles_in_path()

            if obstacle_in_path:
                self.get_logger().warn('Obstacle in path, initiating avoidance...')
                self.avoiding_obstacles = True
                self.execute_obstacle_avoidance()
            else:
                self.avoiding_obstacles = False
                self.follow_path()

            # Update status
            status_msg = String()
            status_msg.data = f"Distance to goal: {dist_to_goal:.2f}m, Obstacles: {len(self.obstacles)}, Active: {self.navigation_active}"
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error in navigation loop: {e}')

    def check_obstacles_in_path(self):
        """Check if there are obstacles blocking the current path"""
        if not self.global_path or not self.obstacles:
            return False

        # Check obstacles near the robot
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        for obs_x, obs_y in self.obstacles:
            # Convert obstacle from robot frame to global frame
            # (In a real implementation, this would use TF transforms)
            global_obs_x = robot_x + obs_x
            global_obs_y = robot_y + obs_y

            # Check if obstacle is near the robot and in the general direction of movement
            obs_dist = math.sqrt((global_obs_x - robot_x)**2 + (global_obs_y - robot_y)**2)

            if obs_dist < self.obstacle_threshold:
                # Calculate angle between robot-to-goal and robot-to-obstacle
                goal_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
                obs_angle = math.atan2(global_obs_y - robot_y, global_obs_x - robot_x)

                # If obstacle is within 45 degrees of goal direction
                angle_diff = abs(goal_angle - obs_angle)
                if angle_diff > math.pi:
                    angle_diff = 2*math.pi - angle_diff

                if angle_diff < math.pi/4:  # Within 45 degrees
                    return True

        return False

    def execute_obstacle_avoidance(self):
        """Execute obstacle avoidance behavior"""
        cmd_vel = Twist()

        # Simple obstacle avoidance: turn away from nearest obstacle
        if self.obstacles:
            # Find nearest obstacle
            nearest_obs = min(self.obstacles, key=lambda obs: math.sqrt(obs[0]**2 + obs[1]**2))
            obs_x, obs_y = nearest_obs

            # Calculate avoidance direction (turn away from obstacle)
            avoidance_angle = math.atan2(obs_y, obs_x) + math.pi  # Opposite direction
            cmd_vel.angular.z = self.angular_speed * math.sin(avoidance_angle)

        # Move forward slowly during avoidance
        cmd_vel.linear.x = self.linear_speed * 0.5

        self.cmd_vel_pub.publish(cmd_vel)

    def follow_path(self):
        """Follow the global path to the goal"""
        cmd_vel = Twist()

        if self.global_path:
            # Simple path following: move toward next waypoint
            next_waypoint = self.global_path[0]  # In a real implementation, find closest point ahead

            # Calculate direction to next waypoint
            dx = next_waypoint.pose.position.x - self.current_pose.position.x
            dy = next_waypoint.pose.position.y - self.current_pose.position.y

            # Calculate distance to waypoint
            dist_to_wp = math.sqrt(dx*dx + dy*dy)

            # Calculate angle to waypoint
            angle_to_wp = math.atan2(dy, dx)

            # Current robot orientation
            robot_yaw = 2 * math.asin(self.current_pose.orientation.z)  # Simplified for z-axis rotation

            # Angular error
            angle_error = angle_to_wp - robot_yaw
            while angle_error > math.pi:
                angle_error -= 2*math.pi
            while angle_error < -math.pi:
                angle_error += 2*math.pi

            # Proportional controller for angular velocity
            cmd_vel.angular.z = 0.5 * angle_error  # Limit angular velocity

            # Linear velocity proportional to progress toward goal
            cmd_vel.linear.x = min(self.linear_speed, max(0.1, dist_to_wp * 0.5))

        self.cmd_vel_pub.publish(cmd_vel)

    def plan_path(self):
        """Plan path from current position to goal using a simple algorithm"""
        if not self.current_pose or not self.current_goal or not self.navigation_active:
            return

        # In a real implementation, this would call Nav2 path planner
        # For this example, we'll use a simple straight-line path with obstacle avoidance

        # Calculate straight-line path
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        goal_x = self.current_goal.position.x
        goal_y = self.current_goal.position.y

        # Create waypoints along straight line
        num_waypoints = max(10, int(math.sqrt((goal_x-start_x)**2 + (goal_y-start_y)**2) / 0.5))

        path = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            wp_x = start_x + t * (goal_x - start_x)
            wp_y = start_y + t * (goal_y - start_y)

            # Create pose for waypoint
            wp_pose = PoseStamped()
            wp_pose.header.stamp = self.get_clock().now().to_msg()
            wp_pose.header.frame_id = 'map'
            wp_pose.pose.position.x = wp_x
            wp_pose.pose.position.y = wp_y
            wp_pose.pose.position.z = 0.0

            path.append(wp_pose)

        self.global_path = path

        # Publish path for visualization
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = [wp for wp in path]

        self.path_pub.publish(path_msg)

    def visualize_navigation_elements(self):
        """Visualize navigation elements in RViz"""
        marker_array = MarkerArray()

        # Visualize current path
        if self.global_path:
            path_marker = Marker()
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.header.frame_id = 'map'
            path_marker.ns = 'navigation_path'
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD

            path_marker.scale.x = 0.05  # Line width
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 0.8

            for wp in self.global_path:
                point = Point()
                point.x = wp.pose.position.x
                point.y = wp.pose.position.y
                point.z = 0.1  # Slightly above ground
                path_marker.points.append(point)

            marker_array.markers.append(path_marker)

        # Visualize obstacles
        for i, (obs_x, obs_y) in enumerate(self.obstacles):
            if math.sqrt(obs_x**2 + obs_y**2) < 5.0:  # Only visualize nearby obstacles
                obs_marker = Marker()
                obs_marker.header.stamp = self.get_clock().now().to_msg()
                obs_marker.header.frame_id = 'base_link'  # Robot frame
                obs_marker.ns = 'obstacles'
                obs_marker.id = i + 100  # Offset to avoid conflicts
                obs_marker.type = Marker.SPHERE
                obs_marker.action = Marker.ADD

                obs_marker.pose.position.x = obs_x
                obs_marker.pose.position.y = obs_y
                obs_marker.pose.position.z = 0.5  # Half a meter high
                obs_marker.pose.orientation.w = 1.0

                obs_marker.scale.x = 0.2
                obs_marker.scale.y = 0.2
                obs_marker.scale.z = 0.2

                obs_marker.color.r = 1.0  # Red for obstacles
                obs_marker.color.g = 0.0
                obs_marker.color.b = 0.0
                obs_marker.color.a = 0.8

                marker_array.markers.append(obs_marker)

        self.debug_markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    nav_node = IsaacNavigationSystem()

    # Example: Set a goal after initialization
    nav_node.set_goal(5.0, 5.0, 0.0)  # Navigate to (5m, 5m)

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        nav_node.get_logger().info('Shutting down Isaac Navigation System...')
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Navigation with Perception Integration

```python
#!/usr/bin/env python3
# perception_navigation_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import numpy as np
from scipy.spatial.distance import cdist
import tf2_ros

class PerceptionNavigationIntegration(Node):
    """
    Integrate perception and navigation systems for robust autonomous navigation
    """

    def __init__(self):
        super().__init__('perception_navigation_integration')

        # Publishers
        self.nav_cmd_pub = self.create_publisher(Twist, '/humanoid/cmd_vel', 10)
        self.nav_status_pub = self.create_publisher(String, '/humanoid/nav_perception/status', 10)

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/humanoid/perception/detections',
            self.detection_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/lidar/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Navigation and perception state
        self.current_pose = None
        self.detected_objects = []
        self.laser_obstacles = []
        self.navigation_goals = []
        self.active_goal = None

        # Integration parameters
        self.object_tracking_horizon = 5.0  # seconds to track objects
        self.dynamic_costmap_inflation = 1.0  # inflation factor for dynamic obstacles
        self.safety_margin = 0.8  # safety margin around detected objects

        # Timers
        self.integration_timer = self.create_timer(0.1, self.integrate_perception_navigation)
        self.safety_timer = self.create_timer(0.05, self.safety_check)

        self.get_logger().info('Perception-Navigation Integration initialized')

    def detection_callback(self, msg):
        """Process perception detections and update navigation awareness"""
        # Update detected objects list
        self.detected_objects = []

        for detection in msg.detections:
            for result in detection.results:
                # Get object position from detection
                # In a real implementation, this would require depth information or stereo triangulation
                # For this example, we'll simulate object positions based on bounding box
                center_x = detection.bbox.center.x
                center_y = detection.bbox.center.y

                # Convert image coordinates to world coordinates (simplified)
                # In reality, this would use depth data and camera calibration
                world_x = center_x * 0.01  # Rough conversion
                world_y = center_y * 0.01  # Rough conversion

                obj_info = {
                    'id': result.id,
                    'score': result.score,
                    'position': (world_x, world_y),
                    'timestamp': self.get_clock().now(),
                    'bbox': detection.bbox
                }

                self.detected_objects.append(obj_info)

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Process laser scan for static obstacle detection
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        self.laser_obstacles = list(zip(ranges[valid_mask], angles[valid_mask]))

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def integrate_perception_navigation(self):
        """Integrate perception data with navigation decisions"""
        if not self.current_pose:
            return

        try:
            # Update navigation based on perceived environment
            self.update_navigation_with_perception()

            # Adjust path planning based on dynamic obstacles
            self.adjust_path_for_dynamic_obstacles()

            # Update costmap with perception data
            self.update_dynamic_costmap()

        except Exception as e:
            self.get_logger().error(f'Error in perception-navigation integration: {e}')

    def update_navigation_with_perception(self):
        """Update navigation behavior based on perception data"""
        # Check for humans or moving objects that require special consideration
        humans_nearby = [obj for obj in self.detected_objects
                        if obj['id'] == 1 and obj['score'] > 0.7]  # Assuming ID 1 is 'person'

        if humans_nearby:
            # Reduce speed and increase safety margins when humans are nearby
            self.get_logger().info(f'Humans detected nearby: {len(humans_nearby)} persons')

            # Increase safety margin for path planning
            self.safety_margin = 1.2  # Increase from 0.8 to 1.2m when humans are nearby

        # Check for other important objects (tables, chairs) for navigation planning
        furniture = [obj for obj in self.detected_objects
                    if obj['id'] in [2, 3] and obj['score'] > 0.6]  # Assuming IDs 2,3 are furniture

        if furniture:
            self.get_logger().info(f'Furniture detected: {len(furniture)} items')

    def adjust_path_for_dynamic_obstacles(self):
        """Adjust planned path to account for dynamic obstacles"""
        # In a real implementation, this would modify the global/local planner
        # to account for detected dynamic obstacles
        # For this example, we'll log the adjustment

        dynamic_obstacles = [obj for obj in self.detected_objects
                            if obj['timestamp'].nanoseconds > (self.get_clock().now().nanoseconds - 5e9)]  # Last 5 seconds

        if dynamic_obstacles:
            self.get_logger().info(f'Adjusting path for {len(dynamic_obstacles)} dynamic obstacles')

    def update_dynamic_costmap(self):
        """Update costmap with perception-derived obstacle information"""
        # This would update the navigation costmap with information from perception system
        # In a real Isaac ROS implementation, this would interface with the Nav2 costmap

        # For this example, we'll just log the activity
        perception_obstacles = len(self.detected_objects)
        laser_obstacles = len(self.laser_obstacles)

        self.get_logger().info(f'Costmap update: {perception_obstacles} perception obstacles, {laser_obstacles} laser obstacles')

    def safety_check(self):
        """Perform safety checks combining perception and navigation data"""
        if not self.current_pose:
            return

        # Check for immediate collision risks
        immediate_risks = []

        # Check laser-based obstacles
        for range_val, angle in self.laser_obstacles:
            if range_val < 0.5:  # Less than 50cm
                # Calculate position of obstacle in world frame
                robot_yaw = 2 * math.asin(self.current_pose.orientation.z)
                world_angle = robot_yaw + angle
                obs_x = self.current_pose.position.x + range_val * math.cos(world_angle)
                obs_y = self.current_pose.position.y + range_val * math.sin(world_angle)

                immediate_risks.append(('laser', range_val, (obs_x, obs_y)))

        # Check perception-based obstacles
        for obj in self.detected_objects:
            if obj['score'] > 0.8:  # High confidence detection
                dist_to_obj = math.sqrt(
                    (obj['position'][0] - self.current_pose.position.x)**2 +
                    (obj['position'][1] - self.current_pose.position.y)**2
                )

                if dist_to_obj < 1.0:  # Within 1 meter
                    immediate_risks.append(('perception', dist_to_obj, obj['position']))

        # Handle immediate risks
        if immediate_risks:
            closest_risk = min(immediate_risks, key=lambda x: x[1])
            risk_type, dist, pos = closest_risk

            self.get_logger().warn(f'IMMEDIATE RISK: {risk_type} obstacle at {dist:.2f}m - stopping navigation')

            # Emergency stop
            cmd_vel = Twist()
            self.nav_cmd_pub.publish(cmd_vel)

    def publish_integration_status(self):
        """Publish status of perception-navigation integration"""
        status_msg = String()
        status_msg.data = f"Perception-Navigation Integration: {len(self.detected_objects)} objects, " \
                         f"{len(self.laser_obstacles)} laser obstacles, " \
                         f"Navigation active: {bool(self.active_goal)}"
        self.nav_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    integration_node = PerceptionNavigationIntegration()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        integration_node.get_logger().info('Shutting down Perception-Navigation Integration...')
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Examples

### Example 1: Isaac ROS Path Planning with Dynamic Obstacle Avoidance

```python
#!/usr/bin/env python3
# dynamic_path_planning.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
import numpy as np
from scipy.spatial.distance import cdist
import math

class DynamicPathPlanner(Node):
    """
    Dynamic path planning with real-time obstacle avoidance using Isaac ROS
    """

    def __init__(self):
        super().__init__('dynamic_path_planner')

        # Publishers
        self.path_cmd_pub = self.create_publisher(PoseStamped, '/humanoid/path_goal', 10)
        self.vel_cmd_pub = self.create_publisher(Twist, '/humanoid/cmd_vel', 10)
        self.debug_pub = self.create_publisher(MarkerArray, '/humanoid/path_planning/debug', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/lidar/scan',
            self.scan_callback,
            10
        )

        # Navigation state
        self.current_pose = None
        self.current_goal = None
        self.waypoints = []
        self.dynamic_obstacles = []
        self.obstacle_velocities = {}  # Track obstacle velocities for prediction

        # Path planning parameters
        self.lookahead_distance = 2.0  # meters ahead to plan
        self.obstacle_buffer = 0.6     # buffer distance around obstacles
        self.max_linear_speed = 0.4    # max linear speed
        self.max_angular_speed = 0.6   # max angular speed

        # Timers
        self.path_update_timer = self.create_timer(0.2, self.update_dynamic_path)
        self.motion_control_timer = self.create_timer(0.1, self.motion_control)

        self.get_logger().info('Dynamic Path Planner initialized')

    def scan_callback(self, msg):
        """Process laser scan to detect and track dynamic obstacles"""
        # Convert scan to obstacle positions
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Convert to Cartesian coordinates relative to robot
        obstacle_x = valid_ranges * np.cos(valid_angles)
        obstacle_y = valid_ranges * np.sin(valid_angles)
        current_obstacles = list(zip(obstacle_x, obstacle_y))

        # Update dynamic obstacle tracking
        self.update_obstacle_tracking(current_obstacles)

    def update_obstacle_tracking(self, current_obstacles):
        """Track obstacles over time to estimate velocities"""
        # For simplicity, we'll just store the current obstacles
        # In a real implementation, we would track obstacles across multiple scans
        # to estimate their velocities and predict future positions
        self.dynamic_obstacles = current_obstacles

    def update_dynamic_path(self):
        """Update path considering dynamic obstacles"""
        if not self.current_goal:
            return

        # Plan path considering current obstacle positions
        # In a real implementation, this would use a dynamic path planning algorithm
        # like Dynamic Window Approach (DWA) or Time Elastic Bands (TEB)

        # For this example, we'll implement a simple reactive approach
        self.reactive_path_adjustment()

    def reactive_path_adjustment(self):
        """Reactive path adjustment based on current obstacles"""
        if not self.current_pose or not self.current_goal:
            return

        # Calculate direct path to goal
        dx = self.current_goal.position.x - self.current_pose.position.x
        dy = self.current_goal.position.y - self.current_pose.position.y
        direct_distance = math.sqrt(dx*dx + dy*dy)

        # Check if path is blocked by obstacles
        if self.check_path_blocked(dx, dy, direct_distance):
            self.get_logger().info('Direct path blocked, calculating alternative route')
            # In a real implementation, this would calculate a new path around obstacles
            # For now, we'll just adjust heading to avoid immediate obstacles
            self.calculate_avoidance_direction()

    def check_path_blocked(self, dx, dy, distance):
        """Check if the direct path to goal is blocked by obstacles"""
        # Simple check: if obstacles are within a cone in front of the robot
        robot_yaw = 2 * math.asin(self.current_pose.orientation.z)

        for obs_x, obs_y in self.dynamic_obstacles:
            # Convert obstacle from robot frame to direction relative to goal
            obs_angle = math.atan2(obs_y, obs_x)
            obs_dist = math.sqrt(obs_x*obs_x + obs_y*obs_y)

            # Check if obstacle is in front of robot and roughly in direction of goal
            angle_diff = abs(obs_angle - robot_yaw)
            if angle_diff > math.pi:
                angle_diff = 2*math.pi - angle_diff

            if obs_dist < 1.0 and angle_diff < math.pi/3:  # Within 1m and 60-degree cone
                return True

        return False

    def calculate_avoidance_direction(self):
        """Calculate direction to avoid obstacles"""
        if not self.dynamic_obstacles:
            return

        # Find the clearest direction to move
        best_direction = 0.0
        max_clear_distance = 0.0

        # Sample different directions
        for angle_offset in np.linspace(-math.pi/2, math.pi/2, 9):
            # Check clearance in this direction
            direction_yaw = 2 * math.asin(self.current_pose.orientation.z) + angle_offset
            test_x = math.cos(direction_yaw)
            test_y = math.sin(direction_yaw)

            # Find closest obstacle in this direction
            min_dist = float('inf')
            for obs_x, obs_y in self.dynamic_obstacles:
                # Project obstacle onto test direction
                proj = obs_x * test_x + obs_y * test_y
                if proj > 0:  # Only consider obstacles ahead
                    perp_dist = abs(obs_x * test_y - obs_y * test_x)  # Perpendicular distance
                    if perp_dist < 0.5:  # Within corridor width
                        dist = math.sqrt(obs_x*obs_x + obs_y*obs_y)
                        min_dist = min(min_dist, dist)

            if min_dist > max_clear_distance:
                max_clear_distance = min_dist
                best_direction = angle_offset

        # Adjust robot heading toward clearest direction
        cmd_vel = Twist()
        cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, best_direction))
        cmd_vel.linear.x = self.max_linear_speed * 0.5  # Move forward at reduced speed

        self.vel_cmd_pub.publish(cmd_vel)

    def motion_control(self):
        """Low-level motion control based on path planning"""
        if not self.current_pose or not self.current_goal:
            return

        # Calculate control commands to follow planned path
        cmd_vel = Twist()

        # Calculate direction to goal
        dx = self.current_goal.position.x - self.current_pose.position.x
        dy = self.current_goal.position.y - self.current_pose.position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)

        if distance_to_goal > 0.2:  # Not at goal
            # Calculate desired heading
            desired_yaw = math.atan2(dy, dx)
            current_yaw = 2 * math.asin(self.current_pose.orientation.z)

            # Calculate angular error
            angle_error = desired_yaw - current_yaw
            while angle_error > math.pi:
                angle_error -= 2*math.pi
            while angle_error < -math.pi:
                angle_error += 2*math.pi

            # Proportional control for angular velocity
            cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, angle_error * 1.0))

            # Move forward if facing approximately the right direction
            if abs(angle_error) < math.pi/4:
                cmd_vel.linear.x = min(self.max_linear_speed, distance_to_goal * 0.5)

        self.vel_cmd_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    planner = DynamicPathPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info('Shutting down Dynamic Path Planner...')
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Navigation Performance Validation

```python
#!/usr/bin/env python3
# navigation_performance_validator.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32
import numpy as np
import math
from collections import deque

class NavigationPerformanceValidator(Node):
    """
    Validate navigation performance metrics for Isaac ROS navigation system
    """

    def __init__(self):
        super().__init__('navigation_performance_validator')

        # Publishers
        self.path_efficiency_pub = self.create_publisher(Float32, '/humanoid/navigation/path_efficiency', 10)
        self.success_rate_pub = self.create_publisher(Float32, '/humanoid/navigation/success_rate', 10)
        self.execution_time_pub = self.create_publisher(Float32, '/humanoid/navigation/execution_time', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/humanoid/planned_path',
            self.path_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/humanoid/goal_pose',
            self.goal_callback,
            10
        )

        # Performance tracking
        self.trajectory_history = deque(maxlen=1000)  # Last 1000 poses
        self.path_history = deque(maxlen=100)        # Last 100 planned paths
        self.goal_history = deque(maxlen=50)         # Last 50 goals
        self.start_time = None
        self.current_goal = None
        self.navigation_start_pose = None

        # Performance metrics
        self.successful_navigations = 0
        self.failed_navigations = 0
        self.total_executions = 0

        # Timers
        self.performance_timer = self.create_timer(1.0, self.calculate_performance_metrics)
        self.metrics_publish_timer = self.create_timer(5.0, self.publish_performance_metrics)

        self.get_logger().info('Navigation Performance Validator initialized')

    def odom_callback(self, msg):
        """Track robot trajectory for performance analysis"""
        self.trajectory_history.append({
            'timestamp': self.get_clock().now(),
            'pose': msg.pose.pose,
            'velocity': msg.twist.twist
        })

    def path_callback(self, msg):
        """Track planned paths for efficiency analysis"""
        self.path_history.append(msg)

    def goal_callback(self, msg):
        """Track navigation goals"""
        self.current_goal = msg.pose
        self.navigation_start_pose = self.get_current_pose()
        self.start_time = self.get_clock().now()

    def get_current_pose(self):
        """Get current pose from trajectory history"""
        if self.trajectory_history:
            return self.trajectory_history[-1]['pose']
        return None

    def calculate_path_efficiency(self):
        """Calculate path efficiency metric"""
        if not self.path_history or not self.trajectory_history:
            return 0.0

        # Get the most recent planned path
        planned_path = self.path_history[-1]
        if not planned_path.poses:
            return 0.0

        # Calculate planned path length
        planned_length = 0.0
        for i in range(1, len(planned_path.poses)):
            p1 = planned_path.poses[i-1].pose.position
            p2 = planned_path.poses[i].pose.position
            dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            planned_length += dist

        # Calculate actual path length (from trajectory)
        if len(self.trajectory_history) < 2:
            return 0.0

        actual_length = 0.0
        for i in range(1, len(self.trajectory_history)):
            p1 = self.trajectory_history[i-1]['pose'].position
            p2 = self.trajectory_history[i]['pose'].position
            dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            actual_length += dist

        # Calculate efficiency (optimal path would be straight line)
        if self.current_goal and self.navigation_start_pose:
            straight_line_dist = math.sqrt(
                (self.current_goal.position.x - self.navigation_start_pose.position.x)**2 +
                (self.current_goal.position.y - self.navigation_start_pose.position.y)**2
            )

            if straight_line_dist > 0:
                optimality_ratio = straight_line_dist / planned_length if planned_length > 0 else 0
                efficiency = min(1.0, optimality_ratio)  # Cap at 1.0
                return efficiency

        return 0.0

    def calculate_success_rate(self):
        """Calculate navigation success rate"""
        total = self.successful_navigations + self.failed_navigations
        if total == 0:
            return 0.0
        return float(self.successful_navigations) / total

    def calculate_execution_time(self):
        """Calculate average navigation execution time"""
        if not self.start_time or not self.current_goal:
            return 0.0

        # Check if goal reached (simplified)
        if self.get_current_pose():
            current_pos = self.get_current_pose().position
            goal_pos = self.current_goal.position
            dist_to_goal = math.sqrt((goal_pos.x - current_pos.x)**2 + (goal_pos.y - current_pos.y)**2)

            if dist_to_goal < 0.5:  # Goal reached
                execution_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                return execution_time

        return 0.0

    def calculate_performance_metrics(self):
        """Calculate all performance metrics"""
        path_efficiency = self.calculate_path_efficiency()
        success_rate = self.calculate_success_rate()
        execution_time = self.calculate_execution_time()

        # Log metrics
        self.get_logger().info(f'Performance Metrics - Path Efficiency: {path_efficiency:.3f}, '
                              f'Success Rate: {success_rate:.3f}, '
                              f'Execution Time: {execution_time:.3f}s')

    def publish_performance_metrics(self):
        """Publish performance metrics to ROS topics"""
        # Publish path efficiency
        efficiency_msg = Float32()
        efficiency_msg.data = self.calculate_path_efficiency()
        self.path_efficiency_pub.publish(efficiency_msg)

        # Publish success rate
        success_msg = Float32()
        success_msg.data = self.calculate_success_rate()
        self.success_rate_pub.publish(success_msg)

        # Publish execution time
        time_msg = Float32()
        time_msg.data = self.calculate_execution_time()
        self.execution_time_pub.publish(time_msg)

def main(args=None):
    rclpy.init(args=args)
    validator = NavigationPerformanceValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down Navigation Performance Validator...')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Navigation and path planning systems are essential for autonomous humanoid robot operation. Isaac ROS provides GPU-accelerated navigation capabilities that integrate perception and planning for robust autonomous navigation. Key components include:

- **Localization**: Accurate pose estimation using multiple sensors and fusion algorithms
- **Mapping**: Construction and maintenance of environment representations
- **Path Planning**: Computation of optimal paths considering static and dynamic obstacles
- **Motion Execution**: Smooth execution of planned trajectories with obstacle avoidance
- **Performance Validation**: Metrics and validation tools to ensure navigation quality

## Exercises

### Conceptual
1. Explain the differences between global path planning and local path planning in robotics navigation systems.

### Logical
1. Analyze the trade-offs between navigation optimality and computational performance in real-time humanoid robot applications. When would you prioritize one over the other?

### Implementation
1. Implement a complete navigation system that integrates perception data with path planning for dynamic obstacle avoidance in a humanoid robot simulation.