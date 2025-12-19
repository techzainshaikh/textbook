---
sidebar_position: 9
title: Chapter 4 - Exercises
description: Exercises for ROS 2 control architecture in humanoid robotics
keywords: [control, architecture, ros2, exercises, humanoid, robotics]
---

# Chapter 4 - Exercises

## Conceptual Exercises

### Exercise 1: Control Architecture Design
**Difficulty**: Intermediate

Design a control architecture for a humanoid robot that can walk, maintain balance, and manipulate objects simultaneously. Identify the different control layers needed, their interactions, and potential conflicts. Explain how you would prioritize these tasks and handle conflicts between them.

**Solution**:
A hierarchical control architecture would include:
- High-level task planner (coordinates walking, balancing, and manipulation)
- Mid-level motion generators (creates trajectories for each task)
- Low-level joint controllers (executes precise joint commands)
Conflicts would be resolved through priority-based allocation of DOFs and coordinated motion planning.

### Exercise 2: Safety System Design
**Difficulty**: Advanced

Design a comprehensive safety system for humanoid robot control that includes emergency stops, error detection, recovery procedures, and safe states. Consider both hardware and software failures, and explain how the system would respond to each type of failure.

**Solution**:
Safety system would include:
- Hardware-level safety (emergency stops, current limits)
- Software-level monitoring (position, velocity, temperature limits)
- Recovery procedures (graceful degradation, safe poses)
- Fail-safe states (emergency stop, passive compliance)

## Logical Exercises

### Exercise 3: Control Priority Logic
**Difficulty**: Intermediate

Design a logical system for determining control priority when multiple controllers want to command the same joint. Consider factors such as safety, task urgency, and coordination requirements. Create a decision tree or algorithm for resolving conflicts.

**Solution**:
Priority algorithm:
1. Safety-related commands (highest priority)
2. Balance maintenance commands
3. Task-level commands (based on urgency)
4. Posture maintenance (lowest priority)
Commands would be blended based on priority and task requirements.

### Exercise 4: Fault Detection Logic
**Difficulty**: Advanced

Create a logical framework for detecting faults in humanoid robot control systems. Define the conditions that constitute a fault, the detection mechanisms, and the classification system for different types of faults. Consider both abrupt and gradual fault onset.

**Solution**:
Fault detection framework:
- Abrupt faults: sudden position/velocity errors, communication timeouts
- Gradual faults: increasing tracking errors, parameter drift
- Detection: statistical analysis, threshold comparisons, model-based prediction
- Classification: hardware, software, communication, environmental

## Implementation Exercises

### Exercise 5: PID Controller Implementation
**Difficulty**: Beginner

Implement a PID controller for a single joint with configurable gains and safety limits. Include anti-windup protection, derivative filtering, and proper initialization. Test the controller with different input profiles and verify its behavior.

```python
# PID Controller Implementation
# WHAT: This code implements a PID controller for a single joint with safety limits and anti-windup
# WHY: To demonstrate fundamental PID control concepts for humanoid robot joints

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import time

class JointPIDController(Node):
    def __init__(self):
        super().__init__('joint_pid_controller')

        # Controller parameters
        # WHAT: Define PID gains and safety limits for the controller
        # WHY: These parameters determine the controller's response and safety behavior
        self.kp = 100.0    # Proportional gain
        self.ki = 0.1      # Integral gain
        self.kd = 10.0     # Derivative gain
        self.max_effort = 100.0  # Maximum effort limit (N-m or appropriate units)
        self.max_integral = 10.0  # Maximum integral term to prevent windup

        # Controller state variables
        # WHAT: Store the current state of the PID controller
        # WHY: These variables are needed for computing the control output
        self.setpoint = 0.0
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.previous_time = self.get_clock().now()

        # Publishers and subscribers
        # WHAT: Create publisher for effort commands and subscriber for joint state feedback
        # WHY: Communication is needed for control and feedback
        self.effort_publisher = self.create_publisher(Float64, '/joint_effort_command', 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_state_feedback', self.joint_state_callback, 10
        )

        # Command subscriber for changing setpoints
        self.setpoint_subscriber = self.create_subscription(
            Float64, '/joint_setpoint', self.setpoint_callback, 10
        )

        # Control timer
        # WHAT: Create a timer to execute the control loop at regular intervals
        # WHY: PID control requires consistent timing for proper derivative calculation
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        self.get_logger().info('Joint PID Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint state from feedback"""
        # WHAT: Process incoming joint state messages to update current position and velocity
        # WHY: Feedback is needed to compute control errors
        if len(msg.position) > 0:
            self.current_position = msg.position[0]
        if len(msg.velocity) > 0:
            self.current_velocity = msg.velocity[0]

    def setpoint_callback(self, msg: Float64):
        """Update the desired setpoint"""
        # WHAT: Process incoming setpoint commands to update the desired position
        # WHY: The controller needs to know the target position to compute errors
        self.setpoint = msg.data
        self.get_logger().info(f'New setpoint received: {self.setpoint}')

    def control_loop(self):
        """Execute the PID control algorithm"""
        # WHAT: This is the main PID control loop that runs at 100Hz
        # WHY: Real-time control requires consistent execution at regular intervals
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        self.previous_time = current_time

        if dt <= 0:  # Avoid division by zero
            return

        # Calculate error
        # WHAT: Compute the difference between desired and actual position
        # WHY: Error is the fundamental input to the PID control algorithm
        error = self.setpoint - self.current_position

        # Update integral term with anti-windup protection
        # WHAT: Accumulate error over time for the integral component
        # WHY: The integral term helps eliminate steady-state error
        self.integral_error += error * dt

        # Anti-windup: limit the integral term to prevent excessive accumulation
        # WHAT: Apply limits to the integral accumulator to prevent windup
        # WHY: Windup occurs when the integral term grows too large during saturation
        self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))

        # Calculate derivative term (using feedback to avoid derivative kick)
        # WHAT: Compute the rate of change of error for the derivative component
        # WHY: The derivative term provides damping to reduce oscillation
        derivative = -self.current_velocity  # Using negative velocity to avoid setpoint derivative

        # Calculate PID output
        # WHAT: Combine the three PID terms to get the control output
        # WHY: The PID formula provides balanced control with proportional, integral, and derivative action
        p_term = self.kp * error
        i_term = self.ki * self.integral_error
        d_term = self.kd * derivative

        effort_output = p_term + i_term + d_term

        # Apply effort limits
        # WHAT: Limit the output to safe values to protect the joint
        # WHY: Excessive effort can damage the motor or transmission
        effort_output = max(-self.max_effort, min(self.max_effort, effort_output))

        # Publish effort command
        # WHAT: Send the computed effort command to the joint
        # WHY: The command needs to be transmitted to the actuator for execution
        effort_msg = Float64()
        effort_msg.data = effort_output
        self.effort_publisher.publish(effort_msg)

        # Store current error for next derivative calculation
        self.previous_error = error

        # Log control information for debugging
        self.get_logger().debug(f'Error: {error:.3f}, P: {p_term:.3f}, I: {i_term:.3f}, D: {d_term:.3f}, Effort: {effort_output:.3f}')

def main(args=None):
    """Main function to initialize and run the PID controller"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    controller = JointPIDController()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the controller when needed
        controller.get_logger().info('Shutting down PID controller')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`

### Exercise 6: Joint Trajectory Controller
**Difficulty**: Intermediate

Implement a joint trajectory controller that interpolates between waypoints and sends smooth commands to the low-level joint controllers. Include proper synchronization between multiple joints and handle trajectory abort conditions.

```python
# Joint Trajectory Controller Implementation
# WHAT: This code implements a trajectory controller that interpolates between waypoints
# WHY: To demonstrate smooth motion control for humanoid robot joints

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import time
import math
from enum import Enum
from typing import List, Dict

class TrajectoryState(Enum):
    """Enumeration of trajectory execution states"""
    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    ABORTED = "aborted"
    COMPLETED = "completed"

class JointTrajectoryController(Node):
    def __init__(self):
        super().__init__('joint_trajectory_controller')

        # Controller parameters
        # WHAT: Define parameters for trajectory execution
        # WHY: These parameters control the smoothness and safety of motion
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]
        self.default_max_velocity = 2.0  # rad/s
        self.default_max_acceleration = 5.0  # rad/s^2

        # Trajectory execution state
        # WHAT: Track the current state of trajectory execution
        # WHY: State management is essential for proper trajectory control
        self.state = TrajectoryState.IDLE
        self.current_trajectory = None
        self.current_point_index = 0
        self.trajectory_start_time = None
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}

        # Publishers and subscribers
        # WHAT: Create communication channels for trajectory control
        # WHY: Communication is needed for receiving trajectories and sending commands
        self.joint_command_publisher = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.trajectory_subscriber = self.create_subscription(
            JointTrajectory, '/joint_trajectory', self.trajectory_callback, 10
        )

        # Timer for trajectory execution
        # WHAT: Create a timer to execute trajectory interpolation at regular intervals
        # WHY: Smooth trajectory execution requires consistent timing
        self.execution_timer = self.create_timer(0.01, self.trajectory_execution_loop)  # 100 Hz

        self.get_logger().info('Joint Trajectory Controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint states from feedback"""
        # WHAT: Process incoming joint state messages to update current positions and velocities
        # WHY: Feedback is needed to track actual joint positions during trajectory execution
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                if i < len(msg.position):
                    self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]

    def trajectory_callback(self, msg: JointTrajectory):
        """Handle incoming trajectory commands"""
        # WHAT: Process incoming trajectory commands and begin execution
        # WHY: The controller needs to know the desired trajectory to follow
        if len(msg.points) == 0:
            self.get_logger().warning('Received trajectory with no points')
            return

        # Validate trajectory joint names
        # WHAT: Verify that the trajectory contains the expected joint names
        # WHY: Invalid joint names would cause the controller to fail
        if msg.joint_names != self.joint_names:
            self.get_logger().error(f'Trajectory joint names mismatch. Expected: {self.joint_names}, Got: {msg.joint_names}')
            return

        # Store trajectory and start execution
        # WHAT: Store the received trajectory and transition to executing state
        # WHY: The controller needs to execute the received trajectory
        self.current_trajectory = msg
        self.current_point_index = 0
        self.trajectory_start_time = self.get_clock().now()
        self.state = TrajectoryState.EXECUTING

        self.get_logger().info(f'Received trajectory with {len(msg.points)} points for joints: {msg.joint_names}')

    def trajectory_execution_loop(self):
        """Execute trajectory interpolation and command generation"""
        # WHAT: This is the main trajectory execution loop that runs at 100Hz
        # WHY: Smooth trajectory execution requires consistent timing for interpolation
        if self.state != TrajectoryState.EXECUTING or self.current_trajectory is None:
            return

        # Check if trajectory is complete
        # WHAT: Verify if all trajectory points have been executed
        # WHY: Execution should stop when the trajectory is complete
        if self.current_point_index >= len(self.current_trajectory.points):
            self.state = TrajectoryState.COMPLETED
            self.get_logger().info('Trajectory execution completed')
            return

        # Get current trajectory point
        # WHAT: Retrieve the current point in the trajectory
        # WHY: The controller needs to interpolate between current and next points
        current_point = self.current_trajectory.points[self.current_point_index]

        # Calculate elapsed time since trajectory start
        # WHAT: Determine how much time has passed since the trajectory began
        # WHY: Time is needed for interpolation between trajectory points
        elapsed_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9

        # Check if we need to advance to the next point
        # WHAT: Determine if enough time has passed to move to the next trajectory point
        # WHY: Trajectories consist of multiple points to be executed over time
        if elapsed_time >= current_point.time_from_start.sec + current_point.time_from_start.nanosec / 1e9:
            self.current_point_index += 1
            if self.current_point_index < len(self.current_trajectory.points):
                # Move to the next point
                next_point = self.current_trajectory.points[self.current_point_index]
                self.trajectory_start_time = self.get_clock().now()
                self.get_logger().info(f'Moved to trajectory point {self.current_point_index}')
            return

        # Calculate interpolation between current and previous points
        # WHAT: Interpolate joint positions between trajectory points
        # WHY: Smooth motion requires interpolation between discrete trajectory points
        if self.current_point_index > 0:
            prev_point = self.current_trajectory.points[self.current_point_index - 1]
            next_point = current_point

            # Calculate interpolation factor
            prev_time = prev_point.time_from_start.sec + prev_point.time_from_start.nanosec / 1e9
            next_time = next_point.time_from_start.sec + next_point.time_from_start.nanosec / 1e9
            current_elapsed = elapsed_time

            # Ensure we don't go beyond the next point's time
            current_elapsed = min(current_elapsed, next_time)

            if next_time > prev_time:
                t = (current_elapsed - prev_time) / (next_time - prev_time)
                t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

                # Linear interpolation between points
                # WHAT: Perform linear interpolation between trajectory points
                # WHY: Linear interpolation provides smooth motion between discrete points
                interpolated_positions = []
                for i in range(len(prev_point.positions)):
                    pos = prev_point.positions[i] + t * (next_point.positions[i] - prev_point.positions[i])
                    interpolated_positions.append(pos)

                # Publish interpolated commands
                # WHAT: Send the interpolated joint positions as commands
                # WHY: Commands need to be sent to the actuators for execution
                command_msg = Float64MultiArray()
                command_msg.data = interpolated_positions
                self.joint_command_publisher.publish(command_msg)

                # Log trajectory progress
                self.get_logger().debug(f'Trajectory progress: {t:.2f}, Positions: {interpolated_positions}')
            else:
                # If times are equal, use the next point directly
                command_msg = Float64MultiArray()
                command_msg.data = next_point.positions
                self.joint_command_publisher.publish(command_msg)
        else:
            # For the first point, publish directly
            command_msg = Float64MultiArray()
            command_msg.data = current_point.positions
            self.joint_command_publisher.publish(command_msg)

    def abort_trajectory(self):
        """Abort current trajectory execution"""
        # WHAT: Stop the current trajectory execution and transition to aborted state
        # WHY: Sometimes trajectories need to be stopped due to safety or other conditions
        if self.state == TrajectoryState.EXECUTING:
            self.state = TrajectoryState.ABORTED
            self.get_logger().warning('Trajectory execution aborted')

    def pause_trajectory(self):
        """Pause current trajectory execution"""
        # WHAT: Pause the current trajectory execution
        # WHY: Trajectories might need to be paused temporarily
        if self.state == TrajectoryState.EXECUTING:
            self.state = TrajectoryState.PAUSED
            self.get_logger().info('Trajectory execution paused')

    def resume_trajectory(self):
        """Resume paused trajectory execution"""
        # WHAT: Resume a paused trajectory execution
        # WHY: Paused trajectories can be resumed after the pause reason is resolved
        if self.state == TrajectoryState.PAUSED:
            self.state = TrajectoryState.EXECUTING
            self.get_logger().info('Trajectory execution resumed')

def main(args=None):
    """Main function to initialize and run the trajectory controller"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    controller = JointTrajectoryController()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the controller when needed
        controller.get_logger().info('Shutting down trajectory controller')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `trajectory_msgs`, `control_msgs`, `std_msgs`

### Exercise 7: Safety Monitor Implementation
**Difficulty**: Advanced

Implement a safety monitor that continuously checks joint limits, velocity limits, temperature limits, and position tracking errors. The monitor should trigger appropriate responses when limits are exceeded, including reducing gains, limiting commands, or initiating emergency stops.

```python
# Safety Monitor Implementation
# WHAT: This code implements a safety monitor for humanoid robot control systems
# WHY: To ensure safe operation with continuous monitoring and emergency procedures

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Temperature
from std_msgs.msg import Bool, String, Float64
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, List

class SafetyLevel(Enum):
    """Enumeration of safety levels"""
    NORMAL = "normal"
    WARNING = "warning"
    ERROR = "error"
    EMERGENCY = "emergency"

class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Safety parameters
        # WHAT: Define safety limits and thresholds for monitoring
        # WHY: Safety parameters determine when the system should take protective actions
        self.position_limits = {
            'hip': (math.radians(-90), math.radians(90)),      # Radians
            'knee': (math.radians(-180), math.radians(0)),     # Radians
            'ankle': (math.radians(-30), math.radians(30)),    # Radians
            'shoulder': (math.radians(-120), math.radians(120)), # Radians
            'elbow': (math.radians(-160), math.radians(0))     # Radians
        }

        self.velocity_limits = {
            'hip': math.radians(120),      # rad/s
            'knee': math.radians(120),     # rad/s
            'ankle': math.radians(180),    # rad/s
            'shoulder': math.radians(120), # rad/s
            'elbow': math.radians(180)     # rad/s
        }

        self.temperature_limits = {
            'hip': 70.0,      # Celsius
            'knee': 70.0,
            'ankle': 70.0,
            'shoulder': 70.0,
            'elbow': 70.0
        }

        self.error_threshold = math.radians(10)  # Maximum tracking error (radians)

        # Joint state tracking
        # WHAT: Store current and previous joint states for monitoring
        # WHY: Monitoring requires comparing current states to limits and previous states
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.joint_temperatures = {}
        self.previous_positions = {}
        self.joint_desired_positions = {}

        # System state
        # WHAT: Track the overall safety state of the system
        # WHY: The safety state determines what actions should be taken
        self.safety_level = SafetyLevel.NORMAL
        self.emergency_active = False
        self.last_emergency_time = None

        # Publishers and subscribers
        # WHAT: Create communication channels for safety monitoring
        # WHY: Communication is needed to receive data and send safety commands
        self.emergency_publisher = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_publisher = self.create_publisher(String, '/safety_status', 10)
        self.error_publisher = self.create_publisher(String, '/error_report', 10)
        self.gain_reduction_publisher = self.create_publisher(Float64, '/gain_reduction', 10)

        # Subscribers for monitoring
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.joint_temp_subscriber = self.create_subscription(
            Temperature, '/joint_temperatures', self.joint_temp_callback, 10
        )

        self.desired_state_subscriber = self.create_subscription(
            JointState, '/joint_desired_states', self.desired_state_callback, 10
        )

        # Safety monitoring timer
        # WHAT: Create a timer to execute safety checks at regular intervals
        # WHY: Continuous monitoring requires consistent timing
        self.safety_timer = self.create_timer(0.1, self.safety_check)  # 10 Hz

        # Initialize joint names
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]

        # Initialize joint dictionaries
        for name in self.joint_names:
            self.joint_positions[name] = 0.0
            self.joint_velocities[name] = 0.0
            self.joint_efforts[name] = 0.0
            self.joint_temperatures[name] = 25.0  # Default temperature
            self.previous_positions[name] = 0.0
            self.joint_desired_positions[name] = 0.0

        self.get_logger().info('Safety Monitor initialized')

    def joint_state_callback(self, msg: JointState):
        """Update joint states from sensor feedback"""
        # WHAT: Process incoming joint state messages to update current joint states
        # WHY: The safety monitor needs current joint states to perform checks
        for i, name in enumerate(msg.name):
            if name in self.joint_positions:
                if i < len(msg.position):
                    self.previous_positions[name] = self.joint_positions[name]
                    self.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.joint_efforts[name] = msg.effort[i]

    def joint_temp_callback(self, msg: Temperature):
        """Update joint temperature information"""
        # WHAT: Process incoming temperature messages to update joint temperatures
        # WHY: Temperature monitoring is important for preventing motor damage
        # This would typically come from a temperature topic, but for simulation
        # we'll use the joint name and effort to estimate temperature
        for joint_name, effort in self.joint_efforts.items():
            # Estimate temperature based on effort (simplified model)
            base_temp = 25.0
            temp_rise = abs(effort) * 0.1
            self.joint_temperatures[joint_name] = base_temp + temp_rise

    def desired_state_callback(self, msg: JointState):
        """Update desired joint positions for error calculation"""
        # WHAT: Process incoming desired state messages to update target positions
        # WHY: Tracking error calculation requires knowledge of desired positions
        for i, name in enumerate(msg.name):
            if name in self.joint_desired_positions and i < len(msg.position):
                self.joint_desired_positions[name] = msg.position[i]

    def check_position_limits(self) -> List[str]:
        """Check joint position limits"""
        # WHAT: Verify that joint positions are within safe limits
        # WHY: Exceeding position limits can cause mechanical damage
        errors = []

        for joint_name, position in self.joint_positions.items():
            joint_type = self.get_joint_type(joint_name)
            limits = self.position_limits.get(joint_type, (-float('inf'), float('inf')))

            if position < limits[0] or position > limits[1]:
                errors.append(f"Joint {joint_name} position limit exceeded: {math.degrees(position):.2f}°, limits: [{math.degrees(limits[0]):.2f}°, {math.degrees(limits[1]):.2f}°]")

        return errors

    def check_velocity_limits(self) -> List[str]:
        """Check joint velocity limits"""
        # WHAT: Verify that joint velocities are within safe limits
        # WHY: Exceeding velocity limits can cause mechanical stress and damage
        errors = []

        for joint_name, velocity in self.joint_velocities.items():
            joint_type = self.get_joint_type(joint_name)
            max_velocity = self.velocity_limits.get(joint_type, float('inf'))

            if abs(velocity) > max_velocity:
                errors.append(f"Joint {joint_name} velocity limit exceeded: {math.degrees(velocity):.2f}°/s, limit: {math.degrees(max_velocity):.2f}°/s")

        return errors

    def check_temperature_limits(self) -> List[str]:
        """Check joint temperature limits"""
        # WHAT: Verify that joint temperatures are within safe limits
        # WHY: Exceeding temperature limits can damage motors and electronics
        errors = []

        for joint_name, temperature in self.joint_temperatures.items():
            joint_type = self.get_joint_type(joint_name)
            max_temp = self.temperature_limits.get(joint_type, 80.0)

            if temperature > max_temp:
                errors.append(f"Joint {joint_name} temperature limit exceeded: {temperature:.2f}°C, limit: {max_temp:.2f}°C")

        return errors

    def check_tracking_errors(self) -> List[str]:
        """Check position tracking errors"""
        # WHAT: Verify that tracking errors are within acceptable limits
        # WHY: Large tracking errors may indicate system problems or instability
        errors = []

        for joint_name, actual_pos in self.joint_positions.items():
            desired_pos = self.joint_desired_positions.get(joint_name, actual_pos)
            error = abs(actual_pos - desired_pos)

            if error > self.error_threshold:
                errors.append(f"Joint {joint_name} tracking error too large: {math.degrees(error):.2f}°, threshold: {math.degrees(self.error_threshold):.2f}°")

        return errors

    def get_joint_type(self, joint_name: str) -> str:
        """Determine joint type from name"""
        # WHAT: Determine the type of joint from its name
        # WHY: Different joint types have different safety limits
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
        """Execute comprehensive safety checks"""
        # WHAT: Perform all safety checks and update system state accordingly
        # WHY: Regular safety checks ensure safe robot operation
        position_errors = self.check_position_limits()
        velocity_errors = self.check_velocity_limits()
        temperature_errors = self.check_temperature_limits()
        tracking_errors = self.check_tracking_errors()

        all_errors = position_errors + velocity_errors + temperature_errors + tracking_errors

        # Update safety level based on errors
        # WHAT: Determine the appropriate safety level based on the errors found
        # WHY: Different safety levels require different responses
        if all_errors:
            # Log all errors
            for error in all_errors:
                self.get_logger().warning(error)

            # Publish error report
            error_msg = String()
            error_msg.data = "; ".join(all_errors)
            self.error_publisher.publish(error_msg)

            # Determine safety level based on error severity
            critical_errors = [err for err in all_errors if 'temperature' in err or 'position' in err]
            if critical_errors:
                if self.safety_level != SafetyLevel.EMERGENCY:
                    self.safety_level = SafetyLevel.EMERGENCY
                    self.emergency_active = True
                    self.last_emergency_time = self.get_clock().now()
                    self.get_logger().error('CRITICAL SAFETY ERROR - Emergency stop initiated')
                    self.publish_emergency_stop()
            else:
                if self.safety_level == SafetyLevel.NORMAL:
                    self.safety_level = SafetyLevel.WARNING
                    self.get_logger().warning('Safety warnings detected')
        else:
            # No errors, check if we can return to normal
            if self.safety_level != SafetyLevel.NORMAL:
                if self.safety_level == SafetyLevel.EMERGENCY:
                    # Check if we can resume from emergency after a delay
                    if (self.get_clock().now() - self.last_emergency_time).nanoseconds / 1e9 > 5.0:
                        self.emergency_active = False
                        self.safety_level = SafetyLevel.NORMAL
                        self.get_logger().info('Safety conditions normalized, resuming operation')
                    else:
                        self.publish_emergency_stop()  # Keep emergency active
                else:
                    self.safety_level = SafetyLevel.NORMAL
                    self.get_logger().info('Safety conditions normalized')

        # Publish safety status
        self.publish_safety_status()

    def publish_emergency_stop(self):
        """Publish emergency stop command"""
        # WHAT: Send an emergency stop command to halt all robot motion
        # WHY: Emergency stop is critical for preventing damage or injury
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_publisher.publish(stop_msg)

    def publish_safety_status(self):
        """Publish current safety status"""
        # WHAT: Publish the current safety status for monitoring
        # WHY: Other nodes and operators need to know the safety state
        status_msg = String()
        status_msg.data = f"Safety Level: {self.safety_level.value}"
        self.safety_status_publisher.publish(status_msg)

def main(args=None):
    """Main function to initialize and run the safety monitor"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    safety_monitor = SafetyMonitor()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the safety monitor when needed
        safety_monitor.get_logger().info('Shutting down safety monitor')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`, `builtin_interfaces`

## Summary

These exercises cover the implementation of control architectures for humanoid robots, from basic PID controllers to advanced trajectory controllers and safety monitoring systems. Each exercise builds on the concepts introduced in the chapter, providing practical implementations with proper error handling and safety considerations.