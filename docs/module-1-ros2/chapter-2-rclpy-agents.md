---
sidebar_position: 3
title: Chapter 2 - rclpy-based Python Agents
description: Developing Python nodes for humanoid robot control using rclpy
keywords: [rclpy, python, ros2, agents, humanoid, control]
---

# Chapter 2 - rclpy-based Python Agents

## Learning Objectives

By the end of this chapter, students will be able to:
1. Create sophisticated Python agents using the rclpy library
2. Implement state machines for humanoid robot behaviors
3. Design agent architectures with proper error handling and recovery
4. Integrate multiple ROS 2 communication patterns within a single agent
5. Apply design patterns for reusable and maintainable robot agents

## Prerequisites

Before starting this chapter, students should have:
- Completed Chapter 1 on Nodes, Topics, Services, and Actions
- Solid understanding of Python programming
- Basic knowledge of object-oriented design patterns
- Understanding of humanoid robot kinematics and control concepts

## Core Concepts

### Agent Architecture

In robotics, an agent is an entity that perceives its environment through sensors and acts upon that environment through actuators. In the context of ROS 2 and humanoid robotics, an agent typically takes the form of a node that implements a specific behavior or function.

A well-designed agent should:
- Have a clear purpose and defined responsibilities
- Respond appropriately to environmental changes
- Maintain internal state to make informed decisions
- Handle errors gracefully and attempt recovery
- Communicate effectively with other agents in the system

### State Management in Agents

Humanoid robots operate in dynamic environments that require complex decision-making. Agents need to maintain state to:
- Track the robot's current situation
- Remember past events and decisions
- Plan future actions based on context
- Coordinate with other agents

### Error Handling and Recovery

Robots must operate reliably in real-world conditions where failures are inevitable. Good agent design includes:
- Comprehensive error detection and logging
- Graceful degradation when components fail
- Recovery strategies for common failure modes
- Safe states that prevent damage to the robot or environment

## Implementation

### Creating a Basic Agent Structure

Let's create a foundational agent structure that can be extended for specific humanoid robot behaviors:

```python
# Example: Base Humanoid Agent
# WHAT: This code creates a base class for humanoid robot agents with common functionality
# WHY: To provide a reusable foundation for implementing specific robot behaviors

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, Any, Optional

class AgentState(Enum):
    """Enumeration of possible agent states"""
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class HumanoidAgent(Node):
    def __init__(self, agent_name: str):
        super().__init__(agent_name)

        # Agent state management
        # WHAT: These variables track the current state of the agent
        # WHY: State management is crucial for handling different operational modes
        self.state = AgentState.IDLE
        self.agent_name = agent_name
        self.last_error: Optional[str] = None
        self.start_time = self.get_clock().now()

        # Publishers for agent status
        # WHAT: These publishers send status and error information to other nodes
        # WHY: Other nodes need to know the agent's state for coordination
        self.status_publisher = self.create_publisher(String, f'/{agent_name}/status', 10)
        self.error_publisher = self.create_publisher(String, f'/{agent_name}/error', 10)

        # Subscriber for joint states (common for humanoid robots)
        # WHAT: This subscribes to joint state messages from the robot
        # WHY: Joint states are needed for many humanoid robot behaviors
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for periodic status updates
        # WHAT: This timer periodically publishes the agent's status
        # WHY: Continuous status updates help with monitoring and debugging
        self.status_timer = self.create_timer(1.0, self.publish_status)

        # Initialize agent-specific components
        # WHAT: This calls a method to initialize subclass-specific components
        # WHY: Allows subclasses to add their own initialization logic
        self._initialize_components()

        self.get_logger().info(f'Agent {agent_name} initialized in {self.state.value} state')

    def _initialize_components(self):
        """Initialize agent-specific components - to be overridden by subclasses"""
        # WHAT: This method is meant to be overridden by subclasses
        # WHY: Allows specific agents to initialize their unique components
        pass

    def joint_state_callback(self, msg: JointState):
        """Process joint state messages"""
        # WHAT: This method processes incoming joint state messages
        # WHY: Joint states provide critical information about the robot's configuration
        # This method can be overridden by subclasses
        pass

    def publish_status(self):
        """Publish current agent status"""
        # WHAT: This method publishes the current agent status
        # WHY: Other nodes and monitoring systems need to know the agent's state
        status_msg = String()
        status_msg.data = f"{self.agent_name}: {self.state.value}"
        self.status_publisher.publish(status_msg)

    def publish_error(self, error_msg: str):
        """Publish error information"""
        # WHAT: This method publishes error information to the error topic
        # WHY: Error information needs to be communicated to monitoring systems
        error_string = String()
        error_string.data = f"{self.agent_name}: {error_msg}"
        self.error_publisher.publish(error_string)
        self.last_error = error_msg

    def transition_state(self, new_state: AgentState):
        """Safely transition between states"""
        # WHAT: This method safely transitions the agent to a new state
        # WHY: Proper state transitions are important for maintaining system consistency
        old_state = self.state
        self.state = new_state
        self.get_logger().info(f'State transition: {old_state.value} -> {new_state.value}')

    def enter_error_state(self, error_msg: str):
        """Enter error state with error message"""
        # WHAT: This method transitions the agent to an error state
        # WHY: Error states indicate problems that need attention or recovery
        self.publish_error(error_msg)
        self.transition_state(AgentState.ERROR)

    def enter_active_state(self):
        """Enter active state"""
        # WHAT: This method transitions the agent to an active state
        # WHY: Active state indicates the agent is performing its primary function
        self.transition_state(AgentState.ACTIVE)

    def enter_idle_state(self):
        """Enter idle state"""
        # WHAT: This method transitions the agent to an idle state
        # WHY: Idle state indicates the agent is ready but not currently active
        self.transition_state(AgentState.IDLE)

    def get_uptime(self) -> float:
        """Get time since agent started"""
        # WHAT: This method calculates the time since the agent started
        # WHY: Uptime information is useful for monitoring and debugging
        current_time = self.get_clock().now()
        duration = current_time - self.start_time
        return duration.nanoseconds / 1e9

class BalanceControllerAgent(HumanoidAgent):
    def __init__(self):
        super().__init__('balance_controller')

        # Balance-specific publishers and subscribers
        # WHAT: These are specific to the balance control functionality
        # WHY: Balance control needs to send velocity commands and publish CoM data
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.com_publisher = self.create_publisher(String, 'center_of_mass', 10)

        # Timer for balance control loop
        # WHAT: This timer runs the balance control algorithm at 20Hz
        # WHY: Balance control needs to run at a high frequency for stability
        self.balance_timer = self.create_timer(0.05, self.balance_control_loop)  # 20Hz

        # Balance control parameters
        # WHAT: These parameters define the balance control thresholds and state
        # WHY: Parameters allow tuning of the balance algorithm
        self.com_threshold = 0.05  # Center of mass threshold in meters
        self.current_com = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.balance_active = False

    def _initialize_components(self):
        """Initialize balance controller specific components"""
        # WHAT: This method initializes components specific to the balance controller
        # WHY: Provides a place to set up balance-specific initialization
        self.get_logger().info('Balance controller initialized')

    def joint_state_callback(self, msg: JointState):
        """Process joint states to calculate center of mass"""
        # WHAT: This method calculates the center of mass from joint positions
        # WHY: Center of mass is critical for balance control
        # Simplified CoM calculation based on joint positions
        # In a real implementation, this would use the robot's URDF and kinematics
        if 'left_foot_joint' in msg.name and 'right_foot_joint' in msg.name:
            left_idx = msg.name.index('left_foot_joint')
            right_idx = msg.name.index('right_foot_joint')

            # Calculate approximate center of mass
            # WHAT: This calculates the X position of the center of mass
            # WHY: The X position is important for forward-back balance
            self.current_com['x'] = (msg.position[left_idx] + msg.position[right_idx]) / 2.0

    def balance_control_loop(self):
        """Main balance control loop"""
        # WHAT: This is the main control loop for maintaining balance
        # WHY: Balance needs to be continuously maintained during operation
        if self.state != AgentState.ACTIVE:
            return

        # Check if center of mass is within safe bounds
        # WHAT: This checks if the center of mass has exceeded the threshold
        # WHY: Excessive CoM deviation indicates potential loss of balance
        if abs(self.current_com['x']) > self.com_threshold:
            self.get_logger().warning(f'Center of mass exceeded threshold: {self.current_com["x"]:.3f}')

            # Generate corrective movement
            # WHAT: This generates a velocity command to correct the CoM position
            # WHY: Corrective movement helps restore balance
            cmd_msg = Twist()
            cmd_msg.linear.x = -0.1 * self.current_com['x']  # Proportional control
            self.cmd_vel_publisher.publish(cmd_msg)

            # Publish CoM info for monitoring
            # WHAT: This publishes CoM information for monitoring purposes
            # WHY: Monitoring systems need to track CoM for analysis
            com_msg = String()
            com_msg.data = f"CoM: x={self.current_com['x']:.3f}, threshold={self.com_threshold:.3f}"
            self.com_publisher.publish(com_msg)

def main(args=None):
    """Main function to initialize and run the balance controller agent"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    balance_agent = BalanceControllerAgent()
    balance_agent.enter_active_state()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(balance_agent)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the agent when needed
        balance_agent.get_logger().info('Shutting down balance controller agent')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        balance_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `std_msgs`, `sensor_msgs`, `geometry_msgs`

### Creating a State Machine Agent

Here's an example of a more complex agent using a state machine pattern:

```python
# Example: Walking State Machine Agent
# WHAT: This code creates a state machine-based agent for humanoid walking control
# WHY: To demonstrate complex behavior management using state machines

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, Any, Optional
import math

class WalkingState(Enum):
    """Walking-specific states"""
    STANDING = "standing"
    PREPARING_STEP = "preparing_step"
    STEPPING = "stepping"
    BALANCING = "balancing"
    EMERGENCY_STOP = "emergency_stop"

class WalkingStateMachineAgent(Node):
    def __init__(self):
        super().__init__('walking_state_machine')

        # State management
        self.current_state = WalkingState.STANDING
        self.previous_state = WalkingState.STANDING
        self.state_start_time = self.get_clock().now()

        # Publishers
        self.joint_cmd_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.state_publisher = self.create_publisher(String, 'walking_state', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10
        )
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, 'cmd_vel_request', self.cmd_vel_request_callback, 10
        )

        # Timers
        self.state_machine_timer = self.create_timer(0.02, self.state_machine_step)  # 50Hz
        self.state_publisher_timer = self.create_timer(0.5, self.publish_state)

        # Walking parameters
        self.step_height = 0.05  # meters
        self.step_length = 0.3   # meters
        self.step_duration = 1.0 # seconds
        self.balance_threshold = 0.1  # meters
        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.joint_positions = {}
        self.cmd_vel_request = Twist()
        self.is_moving = False

        # Walking state variables
        self.step_progress = 0.0
        self.support_foot = 'left'  # 'left' or 'right'
        self.swing_foot = 'right'   # 'left' or 'right'

        self.get_logger().info('Walking State Machine Agent initialized')

    def joint_state_callback(self, msg: JointState):
        """Process joint state messages"""
        for name, position in zip(msg.name, msg.position):
            self.joint_positions[name] = position

    def imu_callback(self, msg: Imu):
        """Process IMU data"""
        # Extract roll, pitch, yaw from quaternion
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # Simplified conversion (assuming small yaw)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y - z * x))

        self.imu_data['roll'] = roll
        self.imu_data['pitch'] = pitch

    def cmd_vel_request_callback(self, msg: Twist):
        """Handle velocity commands"""
        self.cmd_vel_request = msg
        self.is_moving = abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01

    def state_machine_step(self):
        """Main state machine execution step"""
        # Check for emergency conditions first
        if abs(self.imu_data['pitch']) > 0.5 or abs(self.imu_data['roll']) > 0.5:
            if self.current_state != WalkingState.EMERGENCY_STOP:
                self.transition_to_state(WalkingState.EMERGENCY_STOP)
            return

        # Execute current state
        if self.current_state == WalkingState.STANDING:
            self.execute_standing_state()
        elif self.current_state == WalkingState.PREPARING_STEP:
            self.execute_preparing_step_state()
        elif self.current_state == WalkingState.STEPPING:
            self.execute_stepping_state()
        elif self.current_state == WalkingState.BALANCING:
            self.execute_balancing_state()
        elif self.current_state == WalkingState.EMERGENCY_STOP:
            self.execute_emergency_stop_state()

    def execute_standing_state(self):
        """Execute standing state logic"""
        if self.is_moving:
            self.transition_to_state(WalkingState.PREPARING_STEP)
        else:
            # Maintain standing position with minimal adjustments
            self.publish_standing_joints()

    def execute_preparing_step_state(self):
        """Execute step preparation state"""
        # Prepare to shift weight to support foot
        self.step_progress += 0.02  # 50Hz * 0.02 = 1.0 per second

        if self.step_progress >= 1.0:
            self.transition_to_state(WalkingState.STEPPING)
            self.step_progress = 0.0

    def execute_stepping_state(self):
        """Execute actual stepping motion"""
        self.step_progress += 0.02  # 50Hz * 0.02 = 1.0 per step_duration

        if self.step_progress >= 1.0:
            # Step completed, switch support foot
            self.support_foot, self.swing_foot = self.swing_foot, self.support_foot
            self.transition_to_state(WalkingState.BALANCING)
            self.step_progress = 0.0
        else:
            # Execute step trajectory
            self.execute_step_trajectory()

    def execute_balancing_state(self):
        """Execute post-step balancing"""
        self.step_progress += 0.02

        if self.step_progress >= 0.5 or self.is_balance_stable():
            if self.is_moving:
                self.transition_to_state(WalkingState.PREPARING_STEP)
            else:
                self.transition_to_state(WalkingState.STANDING)
            self.step_progress = 0.0

    def execute_emergency_stop_state(self):
        """Execute emergency stop procedures"""
        # Move to safe position
        self.publish_safe_joints()
        self.get_logger().warning('Emergency stop activated - robot is in safe position')

        # Check if it's safe to resume
        if abs(self.imu_data['pitch']) < 0.2 and abs(self.imu_data['roll']) < 0.2:
            self.transition_to_state(WalkingState.STANDING)

    def is_balance_stable(self) -> bool:
        """Check if robot is in stable balance"""
        return abs(self.imu_data['pitch']) < self.balance_threshold and \
               abs(self.imu_data['roll']) < self.balance_threshold

    def execute_step_trajectory(self):
        """Execute the step trajectory for the swing foot"""
        # Calculate step trajectory based on progress
        t = self.step_progress
        height_factor = math.sin(math.pi * t) * self.step_height
        forward_factor = t * self.step_length

        # Generate appropriate joint commands for the step
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
        joint_cmd.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder values

        self.joint_cmd_publisher.publish(joint_cmd)

    def publish_standing_joints(self):
        """Publish joint positions for standing posture"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
        joint_cmd.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Standing position

        self.joint_cmd_publisher.publish(joint_cmd)

    def publish_safe_joints(self):
        """Publish joint positions for safe posture"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
        joint_cmd.position = [0.0, -0.5, 0.5, 0.0, -0.5, 0.5]  # Safe position

        self.joint_cmd_publisher.publish(joint_cmd)

    def transition_to_state(self, new_state: WalkingState):
        """Safely transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = self.get_clock().now()
        self.get_logger().info(f'Walking state transition: {self.previous_state.value} -> {new_state.value}')

        # Publish state change
        state_msg = String()
        state_msg.data = new_state.value
        self.state_publisher.publish(state_msg)

    def publish_state(self):
        """Publish current walking state"""
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    walking_agent = WalkingStateMachineAgent()

    try:
        rclpy.spin(walking_agent)
    except KeyboardInterrupt:
        walking_agent.get_logger().info('Shutting down walking state machine agent')
    finally:
        walking_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `std_msgs`, `sensor_msgs`, `geometry_msgs`

## Examples

### Example: Multi-Agent Coordination

Here's an example of how multiple agents can coordinate in a humanoid robot system:

```python
# Example: Agent Coordinator
# WHAT: This code creates a coordinator agent that manages multiple specialized agents
# WHY: To demonstrate how to coordinate multiple agents for complex humanoid behaviors

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import time
from typing import Dict, Any, List
import threading

class AgentCoordinator(Node):
    def __init__(self):
        super().__init__('agent_coordinator')

        # Publishers
        self.coordinator_status_publisher = self.create_publisher(
            String, 'coordinator_status', 10
        )
        self.system_cmd_publisher = self.create_publisher(
            String, 'system_commands', 10
        )

        # Subscribers
        self.agent_status_subscriber = self.create_subscription(
            String, 'agent_status', self.agent_status_callback, 10
        )
        self.system_control_subscriber = self.create_subscription(
            String, 'system_control', self.system_control_callback, 10
        )

        # Timer for coordination logic
        self.coordination_timer = self.create_timer(0.1, self.coordination_step)

        # Agent management
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_commands: List[str] = []
        self.system_state = "idle"
        self.last_coordination_time = self.get_clock().now()

        # Initialize agent registry
        self.register_default_agents()

        self.get_logger().info('Agent Coordinator initialized')

    def register_default_agents(self):
        """Register known agents in the system"""
        default_agents = [
            "balance_controller",
            "walking_controller",
            "arm_controller",
            "head_controller",
            "navigation_agent"
        ]

        for agent_name in default_agents:
            self.active_agents[agent_name] = {
                "status": "unknown",
                "last_seen": self.get_clock().now(),
                "capabilities": []
            }

    def agent_status_callback(self, msg: String):
        """Process agent status updates"""
        # Parse status message (format: "agent_name: status")
        if ":" in msg.data:
            agent_name, status = msg.data.split(":", 1)
            agent_name = agent_name.strip()
            status = status.strip()

            if agent_name in self.active_agents:
                self.active_agents[agent_name]["status"] = status
                self.active_agents[agent_name]["last_seen"] = self.get_clock().now()
                self.get_logger().debug(f'Updated status for {agent_name}: {status}')
            else:
                # Register new agent if not in registry
                self.active_agents[agent_name] = {
                    "status": status,
                    "last_seen": self.get_clock().now(),
                    "capabilities": []
                }
                self.get_logger().info(f'Registered new agent: {agent_name}')

    def system_control_callback(self, msg: String):
        """Process system control commands"""
        command = msg.data.lower().strip()
        self.agent_commands.append(command)
        self.get_logger().info(f'Received system command: {command}')

    def coordination_step(self):
        """Main coordination logic step"""
        current_time = self.get_clock().now()

        # Check for timed out agents
        timeout_threshold = rclpy.time.Duration(seconds=5.0)
        for agent_name, agent_info in self.active_agents.items():
            time_since_seen = current_time - agent_info["last_seen"]
            if time_since_seen > timeout_threshold:
                self.get_logger().warning(f'Agent {agent_name} appears to be unresponsive')
                # Could trigger recovery procedures here

        # Process any pending commands
        while self.agent_commands:
            command = self.agent_commands.pop(0)
            self.execute_command(command)

        # Publish coordinator status
        status_msg = String()
        active_count = sum(1 for info in self.active_agents.values() if info["status"] != "error")
        status_msg.data = f"Coordinator: {active_count}/{len(self.active_agents)} agents active"
        self.coordinator_status_publisher.publish(status_msg)

    def execute_command(self, command: str):
        """Execute a system command"""
        if command == "start_walking":
            self.send_command_to_agent("walking_controller", "start")
            self.send_command_to_agent("balance_controller", "activate")
        elif command == "stop_walking":
            self.send_command_to_agent("walking_controller", "stop")
            self.send_command_to_agent("balance_controller", "deactivate")
        elif command == "emergency_stop":
            self.emergency_stop_all_agents()
        elif command == "system_check":
            self.perform_system_check()
        else:
            self.get_logger().warning(f'Unknown command: {command}')

    def send_command_to_agent(self, agent_name: str, command: str):
        """Send a command to a specific agent"""
        if agent_name in self.active_agents:
            cmd_msg = String()
            cmd_msg.data = f"{agent_name}: {command}"
            self.system_cmd_publisher.publish(cmd_msg)
            self.get_logger().info(f'Sent command to {agent_name}: {command}')
        else:
            self.get_logger().warning(f'Cannot send command to unknown agent: {agent_name}')

    def emergency_stop_all_agents(self):
        """Send emergency stop to all agents"""
        self.get_logger().warning('Sending emergency stop to all agents')
        for agent_name in self.active_agents.keys():
            self.send_command_to_agent(agent_name, "emergency_stop")

    def perform_system_check(self):
        """Perform a system health check"""
        self.get_logger().info('Performing system health check')

        report = []
        for agent_name, agent_info in self.active_agents.items():
            status = agent_info["status"]
            report.append(f"{agent_name}: {status}")

        self.get_logger().info('System health report: ' + ', '.join(report))

def main(args=None):
    rclpy.init(args=args)
    coordinator = AgentCoordinator()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        coordinator.get_logger().info('Shutting down agent coordinator')
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `std_msgs`, `geometry_msgs`, `sensor_msgs`

## Summary

In this chapter, we've explored the development of sophisticated Python agents for humanoid robot control using rclpy. We covered:

- **Agent Architecture**: Creating reusable base classes for robot agents
- **State Management**: Implementing state machines for complex behaviors
- **Error Handling**: Designing robust error detection and recovery mechanisms
- **Multi-Agent Coordination**: Managing multiple specialized agents

These concepts are crucial for building reliable and maintainable humanoid robot systems. Well-designed agents form the building blocks of complex robotic behaviors and enable modular, scalable robot architectures.

## Exercises

### Conceptual
1. Explain the advantages and disadvantages of using a state machine versus a behavior tree for humanoid robot control.

### Logical
2. Design an agent architecture for a humanoid robot that can walk, pick up objects, and navigate around obstacles. Identify the different agents needed and their interactions.

### Implementation
3. Implement a Python agent that monitors joint temperatures and implements a safety system that reduces motor power when temperatures exceed safe thresholds. Include proper logging and error handling.