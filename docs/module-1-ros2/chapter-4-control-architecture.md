---
sidebar_position: 5
title: Chapter 4 - ROS 2 Control Architecture
description: Designing control systems for humanoid robots using ROS 2
keywords: [ros2, control, humanoid, robotics, architecture, controllers]
---

# Chapter 4 - ROS 2 Control Architecture

## Learning Objectives

By the end of this chapter, students will be able to:
1. Design control architectures for humanoid robots using ROS 2
2. Implement joint controllers with proper feedback and control loops
3. Create high-level controllers for complex humanoid behaviors
4. Integrate perception and control systems for closed-loop operation
5. Design fail-safe mechanisms and error recovery strategies for control systems

## Prerequisites

Before starting this chapter, students should have:
- Understanding of basic control theory (PID, feedback loops)
- Knowledge of ROS 2 communication patterns and message types
- Experience with URDF modeling from Chapter 3
- Understanding of humanoid robot kinematics and dynamics

## Core Concepts

### Control Architecture Overview

Humanoid robot control involves multiple layers:
- **Low-level Joint Control**: Direct motor control and position/velocity/effort control
- **Mid-level Motion Control**: Trajectory generation and interpolation
- **High-level Behavior Control**: Complex behaviors like walking, manipulation, and navigation

### Control Loop Fundamentals

A basic control loop follows the pattern:
1. **Sensing**: Read current state from sensors
2. **Error Calculation**: Compare desired vs. actual state
3. **Control Law**: Apply control algorithm (e.g., PID) to compute command
4. **Actuation**: Send command to actuators
5. **Repeat**: Continuously update based on sensor feedback

For a control system with desired state r(t), actual state y(t), and control output u(t):
- Error: e(t) = r(t) - y(t)
- Control law: u(t) = f(e(t)) (e.g., PID: u(t) = K<sub>p</sub>e(t) + K<sub>i</sub>∫e(t)dt + K<sub>d</sub>de(t)/dt)

### ROS 2 Control Framework

ROS 2 provides the `ros2_control` framework which includes:
- **Hardware Interface**: Abstraction layer for physical hardware
- **Controller Manager**: Runtime management of controllers
- **Controllers**: Specific control algorithms (position, velocity, effort)
- **Joint State Broadcaster**: Publishing joint states for visualization and feedback

### Forward and Inverse Dynamics

For controlling humanoid robots, understanding both:
- **Forward Dynamics**: Given joint torques, compute accelerations
- **Inverse Dynamics**: Given desired motion, compute required torques

## Implementation

### Basic Joint Controller Implementation

Let's create a basic joint controller for humanoid robot joints:

```python
# Example: Joint Position Controller
# WHAT: This code implements a PID-based joint position controller for humanoid robots
# WHY: To demonstrate fundamental joint control with feedback and safety mechanisms

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import numpy as np
import math
from typing import Dict, List, Tuple

class JointPositionController(Node):
    def __init__(self):
        super().__init__('joint_position_controller')

        # Controller parameters
        # WHAT: Define the joint names for the controller
        # WHY: These are the specific joints that will be controlled by this controller
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]

        # PID gains for each joint (tunable parameters)
        # WHAT: Define proportional, integral, and derivative gains for PID control
        # WHY: PID control provides stable and responsive joint position control
        self.kp = {name: 100.0 for name in self.joint_names}  # Proportional gain
        self.ki = {name: 0.1 for name in self.joint_names}    # Integral gain
        self.kd = {name: 10.0 for name in self.joint_names}   # Derivative gain

        # Joint state variables
        # WHAT: Store current and desired positions, velocities, and efforts for each joint
        # WHY: These variables are needed for control calculations and feedback
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.current_efforts = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.commanded_efforts = {name: 0.0 for name in self.joint_names}

        # PID integral and derivative terms
        # WHAT: Store historical values needed for PID calculations
        # WHY: These values are needed to compute the integral and derivative components of PID
        self.integral_errors = {name: 0.0 for name in self.joint_names}
        self.previous_errors = {name: 0.0 for name in self.joint_names}

        # Control loop timing
        # WHAT: Track the time of the last control update
        # WHY: Needed to calculate time delta for integration and differentiation
        self.last_update_time = self.get_clock().now()

        # Publishers and subscribers
        # WHAT: Create publisher for sending joint effort commands
        # WHY: Commands need to be sent to the robot's actuators
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, '/effort_commands', 10
        )

        # WHAT: Create subscriber for receiving current joint states
        # WHY: Feedback is needed to compute control errors
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # WHAT: Create subscriber for receiving trajectory commands
        # WHY: Desired positions come from higher-level trajectory planners
        self.joint_trajectory_subscriber = self.create_subscription(
            JointState, '/joint_trajectory_commands', self.trajectory_command_callback, 10
        )

        # WHAT: Create publisher for controller state monitoring
        # WHY: Other nodes may need to monitor controller performance
        self.controller_state_publisher = self.create_publisher(
            JointTrajectoryControllerState, '/controller_state', 10
        )

        # Control timer (100 Hz)
        # WHAT: Create a timer that calls the control loop at 100 Hz
        # WHY: Real-time control requires consistent timing
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Joint Position Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint states from sensor feedback"""
        # WHAT: Process incoming joint state messages and update current position, velocity, and effort values
        # WHY: The controller needs to know the current state of each joint to compute control errors
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                if i < len(msg.position):
                    self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.current_efforts[name] = msg.effort[i]

    def trajectory_command_callback(self, msg: JointState):
        """Update desired joint positions from trajectory commands"""
        # WHAT: Process incoming trajectory commands and update desired positions
        # WHY: The controller needs to know where each joint should be positioned
        for i, name in enumerate(msg.name):
            if name in self.desired_positions and i < len(msg.position):
                self.desired_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop executing PID control for each joint"""
        # WHAT: This is the main control loop that runs at 100Hz
        # WHY: Real-time control requires consistent execution at regular intervals
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time

        if dt <= 0:  # Avoid division by zero
            # WHAT: Check for valid time delta to avoid division by zero
            # WHY: Division by zero would cause errors in derivative calculation
            return

        # Calculate control commands for each joint
        # WHAT: Compute PID control output for each joint
        # WHY: Each joint needs individual control to achieve desired positions
        for joint_name in self.joint_names:
            # Calculate error
            # WHAT: Compute the difference between desired and actual position
            # WHY: Error is the fundamental input to the control algorithm
            error = self.desired_positions[joint_name] - self.current_positions[joint_name]

            # Update integral term (with anti-windup)
            # WHAT: Accumulate error over time for the integral component
            # WHY: The integral term helps eliminate steady-state error
            self.integral_errors[joint_name] += error * dt
            # Limit integral term to prevent windup
            # WHAT: Apply limits to the integral accumulator to prevent windup
            # WHY: Windup occurs when the integral term grows too large during saturation
            integral_limit = 10.0
            self.integral_errors[joint_name] = max(-integral_limit,
                                                  min(integral_limit,
                                                      self.integral_errors[joint_name]))

            # Calculate derivative term
            # WHAT: Compute the rate of change of error for the derivative component
            # WHY: The derivative term provides damping to reduce oscillation
            if dt > 0:
                derivative = (error - self.previous_errors[joint_name]) / dt
            else:
                derivative = 0.0

            # Store current error for next derivative calculation
            # WHAT: Save the current error for use in the next derivative calculation
            # WHY: Derivative calculation requires the previous error value
            self.previous_errors[joint_name] = error

            # Calculate PID output
            # WHAT: Combine the three PID terms to get the control output
            # WHY: The PID formula provides balanced control with proportional, integral, and derivative action
            p_term = self.kp[joint_name] * error
            i_term = self.ki[joint_name] * self.integral_errors[joint_name]
            d_term = self.kd[joint_name] * derivative

            effort_command = p_term + i_term + d_term

            # Apply safety limits
            # WHAT: Limit the commanded effort to safe values
            # WHY: Protect the robot from excessive forces that could cause damage
            effort_limit = 100.0  # Nm or appropriate unit
            effort_command = max(-effort_limit, min(effort_limit, effort_command))

            # Store commanded effort
            # WHAT: Save the computed effort command for publishing
            # WHY: The effort command needs to be sent to the robot's actuators
            self.commanded_efforts[joint_name] = effort_command

        # Publish effort commands
        # WHAT: Send the computed effort commands to the robot
        # WHY: Commands must be transmitted to the actuators for execution
        command_msg = Float64MultiArray()
        command_msg.data = [self.commanded_efforts[name] for name in self.joint_names]
        self.joint_command_publisher.publish(command_msg)

        # Publish controller state for monitoring
        # WHAT: Publish controller state information for monitoring and debugging
        # WHY: Other nodes may need to monitor controller performance and errors
        state_msg = JointTrajectoryControllerState()
        state_msg.header.stamp = current_time.to_msg()
        state_msg.joint_names = self.joint_names
        state_msg.desired.positions = [self.desired_positions[name] for name in self.joint_names]
        state_msg.actual.positions = [self.current_positions[name] for name in self.joint_names]
        state_msg.error.positions = [self.desired_positions[name] - self.current_positions[name]
                                    for name in self.joint_names]
        self.controller_state_publisher.publish(state_msg)

def main(args=None):
    """Main function to initialize and run the joint position controller"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    controller = JointPositionController()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the controller when needed
        controller.get_logger().info('Shutting down joint position controller')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`, `control_msgs`, `numpy`

### Advanced Control: Whole-Body Controller

Here's an implementation of a more advanced controller for whole-body humanoid control:

```python
# Example: Whole-Body Controller
# WHAT: This code implements a whole-body controller for humanoid robots with multiple tasks
# WHY: To demonstrate hierarchical control for complex humanoid behaviors

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3, Point
from std_msgs.msg import Float64MultiArray, String
from builtin_interfaces.msg import Time
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ControlTask:
    """Represents a control task with priority and weight"""
    name: str
    priority: int  # Lower number = higher priority
    weight: float
    target_value: float
    current_value: float = 0.0

class WholeBodyController(Node):
    def __init__(self):
        super().__init__('whole_body_controller')

        # Robot configuration
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle', 'left_shoulder', 'left_elbow',
            'right_hip', 'right_knee', 'right_ankle', 'right_shoulder', 'right_elbow',
            'torso_yaw', 'torso_pitch', 'neck_yaw', 'neck_pitch'
        ]

        # Joint limits (radians)
        self.joint_limits = {
            name: (-math.pi, math.pi) for name in self.joint_names
        }

        # Initialize joint states
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.current_efforts = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.commanded_torques = {name: 0.0 for name in self.joint_names}

        # Control parameters
        self.control_timestep = 0.001  # 1kHz control
        self.gravity = 9.81  # m/s^2
        self.robot_mass = 50.0  # kg (approximate)

        # Balance control parameters
        self.com_reference = np.array([0.0, 0.0, 0.8])  # Desired CoM position
        self.com_threshold = 0.05  # CoM deviation threshold (m)

        # Walking parameters
        self.step_height = 0.05  # m
        self.step_length = 0.3   # m
        self.step_duration = 1.0 # s

        # IMU data
        self.imu_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.imu_angular_velocity = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.imu_linear_acceleration = np.array([0.0, 0.0, 0.0])  # x, y, z

        # Control tasks
        self.control_tasks: List[ControlTask] = []
        self.initialize_control_tasks()

        # Publishers and subscribers
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, '/effort_commands', 10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.imu_subscriber = self.create_subscription(
            Imu, '/imu_data', self.imu_callback, 10
        )

        self.command_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10
        )

        self.status_publisher = self.create_publisher(String, '/controller_status', 10)

        # Control timer (1kHz)
        self.control_timer = self.create_timer(self.control_timestep, self.control_loop)

        self.get_logger().info('Whole-Body Controller initialized')

    def initialize_control_tasks(self):
        """Initialize the control tasks with priorities and weights"""
        # High priority: Balance maintenance
        self.control_tasks.append(ControlTask(
            name='balance_maintenance',
            priority=0,
            weight=10.0,
            target_value=0.0
        ))

        # Medium priority: Joint position tracking
        self.control_tasks.append(ControlTask(
            name='joint_position_tracking',
            priority=1,
            weight=5.0,
            target_value=0.0
        ))

        # Low priority: Posture maintenance
        self.control_tasks.append(ControlTask(
            name='posture_maintenance',
            priority=2,
            weight=1.0,
            target_value=0.0
        ))

    def joint_state_callback(self, msg: JointState):
        """Update current joint states from sensor feedback"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                if i < len(msg.position):
                    self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.current_efforts[name] = msg.effort[i]

    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        # Orientation (quaternion)
        self.imu_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Angular velocity
        self.imu_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Linear acceleration
        self.imu_linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def command_callback(self, msg: Twist):
        """Handle high-level commands"""
        # For now, just store the command
        # In a real implementation, this would update desired trajectories
        pass

    def calculate_center_of_mass(self) -> np.ndarray:
        """Calculate current center of mass position (simplified)"""
        # This is a simplified calculation - in reality, you'd need the full URDF
        # and mass distribution to calculate CoM accurately
        com_x = 0.0
        com_y = 0.0
        com_z = 0.8  # Approximate CoM height for standing humanoid

        # Add contribution from each joint based on position and estimated mass
        for name, pos in self.current_positions.items():
            if 'hip' in name or 'knee' in name or 'ankle' in name:
                # Legs contribute to CoM position
                if 'left' in name:
                    com_y -= 0.05  # Left leg offset
                elif 'right' in name:
                    com_y += 0.05  # Right leg offset

        return np.array([com_x, com_y, com_z])

    def compute_balance_control(self) -> Dict[str, float]:
        """Compute balance control torques based on CoM position"""
        current_com = self.calculate_center_of_mass()
        com_error = current_com - self.com_reference

        # Simple balance control based on CoM error
        balance_torques = {}

        # Yaw balance (left-right stability)
        if abs(com_error[1]) > self.com_threshold:
            # Apply corrective torques to hip joints
            corrective_torque = -50.0 * com_error[1]  # Proportional control
            balance_torques['left_hip'] = corrective_torque
            balance_torques['right_hip'] = -corrective_torque
        else:
            balance_torques['left_hip'] = 0.0
            balance_torques['right_hip'] = 0.0

        # Pitch balance (forward-back stability)
        if abs(com_error[0]) > self.com_threshold:
            corrective_torque = -30.0 * com_error[0]
            balance_torques['left_ankle'] = corrective_torque
            balance_torques['right_ankle'] = corrective_torque
        else:
            balance_torques['left_ankle'] = 0.0
            balance_torques['right_ankle'] = 0.0

        return balance_torques

    def compute_joint_position_control(self) -> Dict[str, float]:
        """Compute joint position control torques"""
        position_torques = {}

        for name in self.joint_names:
            # Calculate position error
            error = self.desired_positions[name] - self.current_positions[name]

            # Simple PD control
            kp = 50.0  # Position gain
            kd = 5.0   # Velocity gain

            # Calculate torque command
            torque = kp * error - kd * self.current_velocities[name]

            # Apply limits
            torque = max(-100.0, min(100.0, torque))

            position_torques[name] = torque

        return position_torques

    def compute_posture_control(self) -> Dict[str, float]:
        """Compute posture maintenance torques (return to neutral position)"""
        neutral_positions = {
            name: 0.0 for name in self.joint_names
        }

        # Set some default neutral positions
        neutral_positions['left_knee'] = -0.5
        neutral_positions['right_knee'] = -0.5
        neutral_positions['left_ankle'] = 0.5
        neutral_positions['right_ankle'] = 0.5

        posture_torques = {}

        for name in self.joint_names:
            # Calculate error from neutral position
            error = neutral_positions[name] - self.current_positions[name]

            # Simple P control for posture
            posture_torques[name] = 10.0 * error  # Gentle posture maintenance

        return posture_torques

    def hierarchical_control(self) -> Dict[str, float]:
        """Implement hierarchical control prioritizing tasks"""
        # Initialize torques
        total_torques = {name: 0.0 for name in self.joint_names}

        # Sort tasks by priority (lower number = higher priority)
        sorted_tasks = sorted(self.control_tasks, key=lambda x: x.priority)

        # Apply control for each task in priority order
        for task in sorted_tasks:
            if task.name == 'balance_maintenance':
                balance_torques = self.compute_balance_control()
                for joint, torque in balance_torques.items():
                    if joint in total_torques:
                        total_torques[joint] += torque * task.weight
            elif task.name == 'joint_position_tracking':
                position_torques = self.compute_joint_position_control()
                for joint, torque in position_torques.items():
                    if joint in total_torques:
                        total_torques[joint] += torque * task.weight
            elif task.name == 'posture_maintenance':
                posture_torques = self.compute_posture_control()
                for joint, torque in posture_torques.items():
                    if joint in total_torques:
                        total_torques[joint] += torque * task.weight

        return total_torques

    def control_loop(self):
        """Main control loop implementing whole-body control"""
        # Compute control commands using hierarchical approach
        commanded_torques = self.hierarchical_control()

        # Apply commanded torques to joints
        for joint_name in self.joint_names:
            self.commanded_torques[joint_name] = commanded_torques.get(joint_name, 0.0)

        # Publish torque commands
        command_msg = Float64MultiArray()
        command_msg.data = [self.commanded_torques[name] for name in self.joint_names]
        self.joint_command_publisher.publish(command_msg)

        # Publish status
        status_msg = String()
        current_com = self.calculate_center_of_mass()
        status_msg.data = f"CoM: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]"
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = WholeBodyController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down whole-body controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `geometry_msgs`, `std_msgs`, `numpy`

### Safety and Error Handling Controller

Here's an implementation of a safety controller that monitors the system and implements protective measures:

```python
# Example: Safety and Error Handling Controller
# WHAT: This code implements a safety monitoring system for humanoid robot control
# WHY: To ensure safe operation with error detection, recovery, and emergency procedures

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Temperature
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Float64
from builtin_interfaces.msg import Time
import numpy as np
import math
from enum import Enum
from typing import Dict, List, Optional
import time

class SafetyState(Enum):
    """Safety system states"""
    OPERATIONAL = "operational"
    WARNING = "warning"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY = "recovery"

class SafetyController(Node):
    def __init__(self):
        super().__init__('safety_controller')

        # Safety parameters
        self.joint_temp_limits = {
            'hip': 70.0,    # degrees Celsius
            'knee': 70.0,
            'ankle': 70.0,
            'shoulder': 70.0,
            'elbow': 70.0
        }

        self.imu_thresholds = {
            'roll_limit': math.radians(30),    # radians
            'pitch_limit': math.radians(30),   # radians
            'angular_vel_limit': math.radians(5)  # rad/s
        }

        self.joint_effort_limits = {
            'hip': 100.0,    # Nm
            'knee': 80.0,
            'ankle': 60.0,
            'shoulder': 50.0,
            'elbow': 30.0
        }

        # Joint state tracking
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.joint_temperatures = {}

        # IMU data
        self.imu_roll = 0.0
        self.imu_pitch = 0.0
        self.imu_yaw = 0.0
        self.imu_angular_vel = np.array([0.0, 0.0, 0.0])

        # System state
        self.safety_state = SafetyState.OPERATIONAL
        self.last_error_time = None
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

        # Publishers and subscribers
        self.emergency_stop_publisher = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_publisher = self.create_publisher(String, '/safety_status', 10)
        self.error_publisher = self.create_publisher(String, '/error_report', 10)

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.joint_temp_subscriber = self.create_subscription(
            Temperature, '/joint_temperatures', self.joint_temp_callback, 10
        )

        self.imu_subscriber = self.create_subscription(
            Imu, '/imu_data', self.imu_callback, 10
        )

        self.reset_subscriber = self.create_subscription(
            Bool, '/reset_safety', self.reset_callback, 10
        )

        # Safety monitoring timer (10Hz)
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        # Emergency stop timer (for periodic publishing during emergency)
        self.emergency_timer = self.create_timer(0.1, self.emergency_publisher)

        # Initialize as operational
        self.publish_safety_status()
        self.get_logger().info('Safety Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Process joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def joint_temp_callback(self, msg: Temperature):
        """Process joint temperature messages"""
        # This would typically come from a separate temperature topic
        # For this example, we'll simulate it based on joint effort
        for joint_name, effort in self.joint_efforts.items():
            # Simulate temperature based on effort (simplified model)
            base_temp = 25.0  # Base temperature
            temp_rise = abs(effort) * 0.1  # Simplified heating model
            self.joint_temperatures[joint_name] = base_temp + temp_rise

    def imu_callback(self, msg: Imu):
        """Process IMU data"""
        # Convert quaternion to roll/pitch/yaw
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # Simplified conversion (for small yaw angles)
        self.imu_roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self.imu_pitch = math.asin(2 * (w * y - z * x))
        self.imu_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Store angular velocity
        self.imu_angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def reset_callback(self, msg: Bool):
        """Handle safety system reset commands"""
        if msg.data and self.safety_state != SafetyState.OPERATIONAL:
            self.get_logger().info('Safety system reset command received')
            self.safety_state = SafetyState.OPERATIONAL
            self.recovery_attempts = 0
            self.publish_safety_status()

    def check_joint_safety(self) -> List[str]:
        """Check joint safety conditions"""
        errors = []

        for joint_name, effort in self.joint_efforts.items():
            # Check effort limits
            joint_type = self.get_joint_type(joint_name)
            max_effort = self.joint_effort_limits.get(joint_type, 100.0)

            if abs(effort) > max_effort:
                errors.append(f"Joint {joint_name} effort limit exceeded: {effort:.2f} > {max_effort}")

            # Check temperature
            temp = self.joint_temperatures.get(joint_name, 25.0)
            max_temp = self.joint_temp_limits.get(joint_type, 70.0)

            if temp > max_temp:
                errors.append(f"Joint {joint_name} temperature limit exceeded: {temp:.2f} > {max_temp}")

            # Check velocity limits (optional safety check)
            vel = self.joint_velocities.get(joint_name, 0.0)
            max_vel = 5.0  # rad/s (example limit)

            if abs(vel) > max_vel:
                errors.append(f"Joint {joint_name} velocity limit exceeded: {vel:.2f} > {max_vel}")

        return errors

    def check_balance_safety(self) -> List[str]:
        """Check balance and orientation safety"""
        errors = []

        # Check orientation limits
        if abs(self.imu_roll) > self.imu_thresholds['roll_limit']:
            errors.append(f"Roll angle limit exceeded: {math.degrees(self.imu_roll):.2f} > {math.degrees(self.imu_thresholds['roll_limit']):.2f}")

        if abs(self.imu_pitch) > self.imu_thresholds['pitch_limit']:
            errors.append(f"Pitch angle limit exceeded: {math.degrees(self.imu_pitch):.2f} > {math.degrees(self.imu_thresholds['pitch_limit']):.2f}")

        # Check angular velocity limits
        ang_vel_norm = np.linalg.norm(self.imu_angular_vel)
        if ang_vel_norm > self.imu_thresholds['angular_vel_limit']:
            errors.append(f"Angular velocity limit exceeded: {ang_vel_norm:.2f} > {self.imu_thresholds['angular_vel_limit']:.2f}")

        return errors

    def get_joint_type(self, joint_name: str) -> str:
        """Determine joint type from name"""
        if 'hip' in joint_name:
            return 'hip'
        elif 'knee' in joint_name:
            return 'knee'
        elif 'ankle' in joint_name:
            return 'ankle'
        elif 'shoulder' in joint_name:
            return 'shoulder'
        elif 'elbow' in joint_name:
            return 'elbow'
        else:
            return 'hip'  # default

    def safety_check(self):
        """Main safety monitoring function"""
        # Check all safety conditions
        joint_errors = self.check_joint_safety()
        balance_errors = self.check_balance_safety()

        all_errors = joint_errors + balance_errors

        # Update safety state based on errors
        if all_errors:
            # Log all errors
            for error in all_errors:
                self.get_logger().warning(error)

            # Publish error report
            error_msg = String()
            error_msg.data = "; ".join(all_errors)
            self.error_publisher.publish(error_msg)

            # Determine new state based on error severity
            has_critical_error = any(
                "temperature" in err or "angular velocity" in err
                for err in all_errors
            )

            if has_critical_error:
                if self.safety_state != SafetyState.EMERGENCY_STOP:
                    self.last_error_time = time.time()
                    self.safety_state = SafetyState.EMERGENCY_STOP
                    self.get_logger().error('CRITICAL ERROR - Emergency stop activated')
            else:
                if self.safety_state == SafetyState.OPERATIONAL:
                    self.safety_state = SafetyState.WARNING
                    self.get_logger().warning('Safety warnings detected')
        else:
            # No errors, check if we can return to operational
            if self.safety_state == SafetyState.WARNING:
                self.safety_state = SafetyState.OPERATIONAL
                self.get_logger().info('Safety conditions normalized')

        # Handle recovery logic
        if self.safety_state == SafetyState.EMERGENCY_STOP:
            if self.recovery_attempts < self.max_recovery_attempts:
                # Attempt recovery after a delay
                if (time.time() - self.last_error_time) > 5.0:  # 5 second delay
                    self.safety_state = SafetyState.RECOVERY
                    self.get_logger().info(f'Attempting recovery (attempt {self.recovery_attempts + 1})')
                    self.recovery_attempts += 1
            else:
                self.get_logger().error('Maximum recovery attempts reached. Manual reset required.')

        # Publish safety status
        self.publish_safety_status()

    def publish_safety_status(self):
        """Publish current safety status"""
        status_msg = String()
        status_msg.data = f"State: {self.safety_state.value}"

        if self.safety_state == SafetyState.EMERGENCY_STOP:
            status_msg.data += f" | Recovery attempts: {self.recovery_attempts}/{self.max_recovery_attempts}"

        self.safety_status_publisher.publish(status_msg)

    def emergency_publisher(self):
        """Publish emergency stop commands when in emergency state"""
        if self.safety_state == SafetyState.EMERGENCY_STOP:
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_publisher.publish(stop_msg)
        else:
            # When not in emergency, publish false to allow normal operation
            if self.safety_state != SafetyState.RECOVERY:
                stop_msg = Bool()
                stop_msg.data = False
                self.emergency_stop_publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    safety_controller = SafetyController()

    try:
        rclpy.spin(safety_controller)
    except KeyboardInterrupt:
        safety_controller.get_logger().info('Shutting down safety controller')
    finally:
        safety_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `geometry_msgs`, `std_msgs`, `numpy`

## Examples

### Example: Walking Controller with Trajectory Generation

Here's a complete example that combines trajectory generation with walking control:

```python
# Example: Walking Controller with Trajectory Generation
# WHAT: This code implements a walking controller with foot trajectory generation
# WHY: To demonstrate how to generate and execute walking patterns for humanoid robots

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Point, Vector3
from std_msgs.msg import Float64MultiArray, String, Bool
from builtin_interfaces.msg import Time
import numpy as np
import math
from typing import Dict, List, Tuple
from enum import Enum

class WalkingState(Enum):
    """Walking controller states"""
    IDLE = "idle"
    STARTING = "starting"
    WALKING = "walking"
    STOPPING = "stopping"
    BALANCING = "balancing"

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Walking parameters
        self.step_height = 0.05  # m
        self.step_length = 0.3   # m
        self.step_duration = 1.0 # s
        self.zmp_margin = 0.02   # Zero Moment Point safety margin (m)

        # Robot configuration
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
        ]

        # Walking state
        self.walking_state = WalkingState.IDLE
        self.walk_velocity = Twist()  # Linear and angular velocity commands
        self.support_foot = 'left'  # Which foot is currently supporting weight
        self.swing_foot = 'right'   # Which foot is swinging
        self.step_progress = 0.0    # Progress in current step (0.0 to 1.0)
        self.step_count = 0         # Number of completed steps

        # Joint states
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.commanded_torques = {name: 0.0 for name in self.joint_names}

        # IMU data for balance
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Publishers and subscribers
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, '/effort_commands', 10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.imu_subscriber = self.create_subscription(
            Imu, '/imu_data', self.imu_callback, 10
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        self.status_publisher = self.create_publisher(String, '/walking_status', 10)
        self.step_publisher = self.create_publisher(Bool, '/step_event', 10)

        # Control timer (200Hz for walking control)
        self.control_timer = self.create_timer(0.005, self.control_loop)

        self.get_logger().info('Walking Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions and i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        # Convert quaternion to roll/pitch/yaw (simplified)
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        self.roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self.pitch = math.asin(2 * (w * y - z * x))

    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands"""
        self.walk_velocity = msg

        # Change walking state based on command
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            if self.walking_state == WalkingState.IDLE:
                self.walking_state = WalkingState.STARTING
        else:
            if self.walking_state == WalkingState.WALKING:
                self.walking_state = WalkingState.STOPPING

    def generate_foot_trajectory(self, progress: float, foot: str) -> Tuple[float, float, float]:
        """Generate 3D trajectory for a foot during walking"""
        # Simplified trajectory generation
        # X: Forward/backward movement
        x = progress * self.step_length if foot == self.swing_foot else 0.0

        # Y: Lateral movement (for balance)
        y = -0.1 if foot == 'left' else 0.1  # Fixed stance width

        # Z: Vertical movement (foot lifting)
        if foot == self.swing_foot:
            # Lift foot in a parabolic trajectory
            lift_factor = math.sin(math.pi * progress)
            z = self.step_height * lift_factor
        else:
            # Support foot stays on ground
            z = 0.0

        return x, y, z

    def compute_inverse_kinematics(self, target_pos: Tuple[float, float, float], foot: str) -> Dict[str, float]:
        """Compute inverse kinematics for 3-DOF leg to reach target position"""
        x, y, z = target_pos

        # Simplified 3-DOF inverse kinematics for humanoid leg
        # This is a basic implementation - real robots would use more sophisticated IK

        # Calculate leg length requirements
        hip_to_foot_xz = math.sqrt(x*x + z*z)  # Horizontal distance from hip to foot
        leg_length = 0.4  # Simplified leg length (thigh + shin)

        # Check if target is reachable
        if hip_to_foot_xz > leg_length:
            # Scale down to reachable position
            scale = leg_length / hip_to_foot_xz
            x *= scale
            z *= scale

        # Calculate knee angle (using law of cosines)
        # For a simplified 2D leg (sagittal plane)
        d = math.sqrt(x*x + z*z)  # Distance from hip to target

        if d > 0.01:  # Avoid division by zero
            # Knee angle calculation
            cos_knee = (0.2*0.2 + 0.2*0.2 - d*d) / (2 * 0.2 * 0.2)  # Assuming thigh=shin=0.2m
            cos_knee = max(-1.0, min(1.0, cos_knee))  # Clamp to valid range
            knee_angle = math.pi - math.acos(cos_knee)

            # Hip angles
            hip_pitch = math.atan2(x, z) - math.asin((0.2 * math.sin(math.pi - knee_angle)) / d)
            hip_roll = math.atan2(y, max(abs(x), 0.01)) * 0.5  # Simplified lateral adjustment
        else:
            # Default standing position
            knee_angle = 0.0
            hip_pitch = 0.0
            hip_roll = 0.0

        # Ankle angle for foot level
        ankle_angle = -hip_pitch  # Simplified compensation

        # Create joint angle dictionary based on foot
        if foot == 'left':
            return {
                'left_hip': hip_roll,      # Simplified - would be hip_yaw, hip_roll, hip_pitch
                'left_knee': knee_angle,
                'left_ankle': ankle_angle
            }
        else:  # right foot
            return {
                'right_hip': -hip_roll,    # Opposite for right leg
                'right_knee': knee_angle,
                'right_ankle': ankle_angle
            }

    def balance_control(self) -> Dict[str, float]:
        """Compute balance control adjustments based on IMU data"""
        balance_adjustments = {}

        # Simple balance control based on roll/pitch
        max_adjustment = 0.1  # rad

        # Adjust ankle angles to counteract roll
        roll_correction = -self.roll * 2.0  # Gain for roll correction
        roll_correction = max(-max_adjustment, min(max_adjustment, roll_correction))

        # Adjust hip angles to counteract pitch
        pitch_correction = -self.pitch * 1.5  # Gain for pitch correction
        pitch_correction = max(-max_adjustment, min(max_adjustment, pitch_correction))

        balance_adjustments.update({
            'left_ankle': roll_correction,
            'right_ankle': -roll_correction,  # Opposite for right foot
            'left_hip': pitch_correction,
            'right_hip': pitch_correction
        })

        return balance_adjustments

    def control_loop(self):
        """Main walking control loop"""
        # Update walking state machine
        self.update_walking_state()

        # Generate walking pattern based on current state
        if self.walking_state in [WalkingState.WALKING, WalkingState.STARTING]:
            self.execute_walking_step()
        elif self.walking_state == WalkingState.BALANCING:
            self.execute_balancing()
        else:
            # In IDLE, STOPPING states - maintain standing position
            self.maintain_standing_position()

        # Apply balance corrections
        balance_adjustments = self.balance_control()

        # Calculate final joint positions
        final_positions = {}
        for joint in self.joint_names:
            base_pos = self.desired_positions[joint]
            adjustment = balance_adjustments.get(joint, 0.0)
            final_positions[joint] = base_pos + adjustment

        # Publish commands
        command_msg = Float64MultiArray()
        command_msg.data = [final_positions[name] for name in self.joint_names]
        self.joint_command_publisher.publish(command_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.walking_state.value}, Progress: {self.step_progress:.2f}, Support: {self.support_foot}"
        self.status_publisher.publish(status_msg)

    def update_walking_state(self):
        """Update the walking state machine"""
        if self.walking_state == WalkingState.STARTING:
            # Transition to walking after initial setup
            self.walking_state = WalkingState.WALKING
            self.step_progress = 0.0
        elif self.walking_state == WalkingState.WALKING:
            # Check if we need to transition to stopping
            if abs(self.walk_velocity.linear.x) < 0.01 and abs(self.walk_velocity.angular.z) < 0.01:
                if self.step_progress > 0.8:  # Wait for step completion
                    self.walking_state = WalkingState.STOPPING
        elif self.walking_state == WalkingState.STOPPING:
            # Check if stopping is complete
            if self.step_progress >= 1.0:
                self.walking_state = WalkingState.IDLE
        elif self.walking_state == WalkingState.IDLE:
            # Check if we need to start walking
            if abs(self.walk_velocity.linear.x) > 0.01 or abs(self.walk_velocity.angular.z) > 0.01:
                self.walking_state = WalkingState.STARTING

    def execute_walking_step(self):
        """Execute a single walking step"""
        # Update step progress
        step_increment = 0.01  # Adjust based on control frequency and desired step duration
        self.step_progress += step_increment / self.step_duration

        if self.step_progress >= 1.0:
            # Step completed, switch support foot
            self.support_foot, self.swing_foot = self.swing_foot, self.support_foot
            self.step_progress = 0.0
            self.step_count += 1

            # Publish step event
            step_msg = Bool()
            step_msg.data = True
            self.step_publisher.publish(step_msg)

        # Generate trajectories for both feet
        support_pos = self.generate_foot_trajectory(0.0, self.support_foot)  # Support foot stays put
        swing_pos = self.generate_foot_trajectory(self.step_progress, self.swing_foot)  # Swing foot moves

        # Compute joint angles for swing foot
        swing_angles = self.compute_inverse_kinematics(swing_pos, self.swing_foot)

        # For support foot, maintain stable position or slight adjustment
        support_angles = self.compute_inverse_kinematics(support_pos, self.support_foot)

        # Update desired positions
        for joint, angle in swing_angles.items():
            self.desired_positions[joint] = angle

        for joint, angle in support_angles.items():
            self.desired_positions[joint] = angle

    def execute_balancing(self):
        """Execute balancing behavior"""
        # For now, just maintain current position with balance adjustments
        pass

    def maintain_standing_position(self):
        """Maintain a standing position"""
        standing_angles = {
            'left_hip': 0.0,
            'left_knee': -0.5,  # Bent knee for stability
            'left_ankle': 0.5,   # Compensate for bent knee
            'right_hip': 0.0,
            'right_knee': -0.5,
            'right_ankle': 0.5
        }

        for joint, angle in standing_angles.items():
            self.desired_positions[joint] = angle

def main(args=None):
    rclpy.init(args=args)
    walking_controller = WalkingController()

    try:
        rclpy.spin(walking_controller)
    except KeyboardInterrupt:
        walking_controller.get_logger().info('Shutting down walking controller')
    finally:
        walking_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `geometry_msgs`, `std_msgs`, `numpy`

## Summary

In this chapter, we've explored ROS 2 control architecture for humanoid robots:
- **Basic Joint Control**: PID controllers for individual joints with safety limits
- **Whole-Body Control**: Hierarchical control systems managing multiple tasks simultaneously
- **Safety Systems**: Error detection, emergency procedures, and recovery mechanisms
- **Walking Control**: Trajectory generation and gait control for bipedal locomotion

The control architecture forms the backbone of humanoid robot operation, requiring careful consideration of stability, safety, and coordination between multiple subsystems. Proper control design ensures reliable and safe robot operation.

## Exercises

### Conceptual
1. Explain the differences between position, velocity, and effort control in humanoid robots. When would you use each type?

### Logical
2. Design a control architecture for a humanoid robot that can walk, maintain balance, and manipulate objects simultaneously. How would you prioritize these tasks and handle conflicts?

### Implementation
3. Implement a Python controller that uses inverse kinematics to make a humanoid robot arm follow a Cartesian trajectory while avoiding joint limits. Include proper error handling for unreachable positions.