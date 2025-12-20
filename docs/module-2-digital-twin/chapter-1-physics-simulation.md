---
title: Physics Simulation in Gazebo
sidebar_position: 2
description: Understanding physics simulation for digital twins using Gazebo
keywords: [physics simulation, gazebo, robotics simulation, collision detection]
---

# Chapter 1: Physics Simulation in Gazebo

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain the fundamental physics principles underlying robot simulation
- Configure Gazebo simulation environments with accurate physics parameters
- Implement collision detection and response mechanisms
- Tune physics parameters for realistic robot behavior in simulation
- Compare simulation results with real-world robot performance

## Prerequisites

Students should have:
- Understanding of basic physics concepts (gravity, friction, momentum)
- Basic knowledge of ROS 2 communication (covered in Module 1)
- Familiarity with coordinate systems and transformations

## Core Concepts

Physics simulation in robotics relies on numerical integration of differential equations that describe motion. The key components include:

### Newtonian Mechanics in Simulation

The simulation engine calculates forces acting on each rigid body and integrates Newton's equations of motion:
$$F = ma$$
$$\tau = I\alpha$$

Where:
- $F$ is the net force applied to a body
- $m$ is the mass of the body
- $a$ is the resulting acceleration
- $\tau$ is the torque applied
- $I$ is the moment of inertia
- $\alpha$ is the angular acceleration

### Collision Detection and Response

Simulation engines use various algorithms to detect and respond to collisions:
- Broad phase: Quick elimination of non-colliding pairs
- Narrow phase: Precise collision detection between potentially colliding objects
- Contact resolution: Calculation of forces to prevent penetration

### Numerical Integration

Simulators use numerical methods to approximate the continuous motion of bodies:
- Euler integration (simple but unstable)
- Runge-Kutta methods (more accurate)
- Verlet integration (stable for constraints)

## Implementation

Let's implement a basic physics simulation using Gazebo. We'll create a simple humanoid robot model and configure its physical properties.

### Setting Up Gazebo Environment

First, let's create a launch file to start Gazebo with our custom world:

```xml
<!-- launch/humanoid_sim.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(
        get_package_share_directory('humanoid_simulation'),
        'worlds',
        'humanoid_world.world'
    )

    return LaunchDescription([
        # Launch Gazebo with our world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={
                'world': world_file,
                'verbose': 'true'
            }.items()
        ),

        # Spawn our robot in the simulation
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'humanoid_robot',
                '-file', os.path.join(get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.urdf'),
                '-x', '0', '-y', '0', '-z', '1.0'
            ],
            output='screen'
        )
    ])
```

### Physics Properties Configuration

The physical properties of our robot are defined in the URDF file with SDF extensions for Gazebo-specific parameters:

```xml
<!-- URDF snippet with Gazebo physics -->
<link name="torso">
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <inertia ixx="0.5" ixy="0.0" ixz="0.0"
             iyy="0.5" iyz="0.0"
             izz="0.5"/>
  </inertial>

  <visual>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/torso.stl"/>
    </geometry>
  </visual>

  <collision>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/torso_collision.stl"/>
    </geometry>
  </collision>
</link>

<gazebo reference="torso">
  <mu1>0.8</mu1>  <!-- Friction coefficient -->
  <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>      <!-- Contact damping -->
  <material>Gazebo/Orange</material>
</gazebo>
```

### Physics Parameter Tuning

Different scenarios require different physics parameters:

**High Precision Applications (Manipulation):**
- Small time steps (1ms or less)
- High solver iterations (100+)
- Accurate collision geometry

**Fast Prototyping:**
- Larger time steps (5-10ms)
- Lower solver iterations (50-100)
- Simplified collision geometry

## Examples

### Example 1: Gravity Compensation Test

Let's create a simple test to verify our physics simulation is working correctly:

```python
#!/usr/bin/env python3
# test_gravity_compensation.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class GravityCompensationTest(Node):
    """Test gravity compensation in simulation"""

    def __init__(self):
        super().__init__('gravity_compensation_test')

        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer to send commands
        self.timer = self.create_timer(0.1, self.test_callback)
        self.joint_names = ['hip_joint', 'knee_joint', 'ankle_joint']

        self.get_logger().info('Gravity compensation test node started')

    def test_callback(self):
        """Send zero torques and observe drift due to imperfect simulation"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [0.0, 0.0, 0.0]  # Zero torque command
        self.joint_cmd_pub.publish(cmd_msg)

    def joint_state_callback(self, msg):
        """Process joint state feedback"""
        # Log positions to observe drift over time
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                self.get_logger().info(f'{name}: {pos:.4f}')

def main(args=None):
    rclpy.init(args=args)
    test_node = GravityCompensationTest()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Collision Detection Validation

```python
#!/usr/bin/env python3
# collision_validation.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class CollisionValidator(Node):
    """Validate collision detection by monitoring distance to obstacles"""

    def __init__(self):
        super().__init__('collision_validator')

        # TF listener to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers for visualization
        self.marker_pub = self.create_publisher(Marker, 'collision_distance', 10)
        self.safety_pub = self.create_publisher(PointStamped, 'safety_zone', 10)

        # Timer to check distances
        self.timer = self.create_timer(0.5, self.check_distances)

        # Define obstacle positions (should match your world file)
        self.obstacles = [
            {'name': 'wall1', 'position': [2.0, 0.0, 0.0]},
            {'name': 'obstacle1', 'position': [1.0, 1.0, 0.0]}
        ]

        self.robot_frame = 'base_link'
        self.world_frame = 'world'

    def check_distances(self):
        """Check distance from robot to obstacles"""
        try:
            # Get robot transform
            t = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            robot_pos = [t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z]

            min_distance = float('inf')
            closest_obstacle = None

            for obs in self.obstacles:
                dist = np.linalg.norm(np.array(robot_pos[:2]) - np.array(obs['position'][:2]))
                if dist < min_distance:
                    min_distance = dist
                    closest_obstacle = obs['name']

            # Log safety information
            if min_distance < 0.5:  # Less than 50cm to obstacle
                self.get_logger().warn(f'CLOSE TO OBSTACLE: {closest_obstacle}, distance: {min_distance:.2f}m')
            elif min_distance < 1.0:
                self.get_logger().info(f'NEAR OBSTACLE: {closest_obstacle}, distance: {min_distance:.2f}m')
            else:
                self.get_logger().info(f'SAFE DISTANCE: {min_distance:.2f}m from {closest_obstacle}')

        except Exception as e:
            self.get_logger().error(f'Transform lookup failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    validator = CollisionValidator()

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

## Summary

Physics simulation forms the foundation of digital twin technology in robotics. Properly configured physics engines like Gazebo enable safe testing of robot behaviors before real-world deployment. Key considerations include:

- **Accuracy vs. Performance**: Balance realistic physics with computational efficiency
- **Parameter Tuning**: Adjust friction, damping, and stiffness to match real-world behavior
- **Validation**: Continuously compare simulation results with real robot performance
- **Safety**: Use simulation to test dangerous scenarios without risk to equipment or personnel

## Exercises

### Conceptual
1. Explain why physics simulation is crucial for robotics development and how it relates to the concept of digital twins.

### Logical
1. Analyze the trade-offs between simulation accuracy and computational performance. When would you prioritize one over the other?

### Implementation
1. Create a Gazebo simulation with a humanoid robot and implement a simple physics validation test that measures how closely the simulated robot behaves to theoretical physics predictions.