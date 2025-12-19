---
sidebar_position: 7
title: Chapter 2 - Exercises
description: Exercises for rclpy-based Python agents in humanoid robotics
keywords: [rclpy, python, agents, exercises, humanoid, control]
---

# Chapter 2 - Exercises

## Conceptual Exercises

### Exercise 1: Agent Architecture Design
**Difficulty**: Intermediate

Design an agent architecture for a humanoid robot's walking controller. Identify the different agents needed, their responsibilities, and how they communicate with each other. Consider state management, error handling, and recovery mechanisms. Draw a diagram showing the agent interactions.

**Solution**:
A walking controller architecture might include:
- Balance controller agent (maintains stability)
- Step planner agent (plans foot placement)
- Trajectory generator agent (generates joint trajectories)
- State estimator agent (tracks current state)
- Safety monitor agent (detects and handles errors)

### Exercise 2: State Machine Design
**Difficulty**: Advanced

Design a state machine for a humanoid robot's walking behavior. Identify the different states, transitions, and triggers. Consider how to handle disturbances, emergencies, and transitions between different walking gaits (standing, walking, turning, stopping).

**Solution**:
States: STANDING, PREPARING_STEP, STEPPING, BALANCING, EMERGENCY_STOP
Transitions triggered by: sensor data, commands, error conditions, completion of actions

## Logical Exercises

### Exercise 3: Agent Coordination Logic
**Difficulty**: Intermediate

Implement the logic for coordinating multiple agents in a humanoid robot system. Design how agents share information, handle conflicts, and maintain consistency. Consider priority-based conflict resolution and failover mechanisms.

**Solution**:
Coordination logic includes: shared state management, priority-based command arbitration, heartbeat monitoring, and graceful degradation when agents fail.

### Exercise 4: Error Propagation Analysis
**Difficulty**: Advanced

Analyze how errors in one agent can propagate through a multi-agent system. Design isolation mechanisms and error containment strategies to prevent cascading failures. Consider both software and hardware failure scenarios.

**Solution**:
Error isolation includes: agent boundaries with timeouts, circuit breakers, graceful degradation, and fallback behaviors when components fail.

## Implementation Exercises

### Exercise 5: Basic Agent Implementation
**Difficulty**: Beginner

Implement a basic agent that monitors joint temperatures and implements a safety system that reduces motor power when temperatures exceed safe thresholds. Include proper logging and error handling.

```python
# Basic Agent Implementation
# WHAT: This code creates an agent that monitors joint temperatures and implements safety controls
# WHY: To demonstrate basic agent concepts with monitoring, safety controls, and error handling

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Temperature
from std_msgs.msg import Float64MultiArray, String
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, List

class SafetyState(Enum):
    """Enumeration of safety states"""
    SAFE = "safe"
    WARNING = "warning"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

class JointTemperatureMonitorAgent(Node):
    def __init__(self):
        super().__init__('joint_temperature_monitor_agent')

        # Agent state management
        # WHAT: These variables track the current safety state of the agent
        # WHY: State management is crucial for handling different safety levels
        self.safety_state = SafetyState.SAFE
        self.last_error_time = None

        # Joint temperature storage
        # WHAT: Store current temperatures for each joint
        # WHY: Temperatures need to be monitored for safety
        self.joint_temperatures = {}
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow',
            'right_shoulder', 'right_elbow'
        ]

        # Initialize temperatures
        for joint_name in self.joint_names:
            self.joint_temperatures[joint_name] = 25.0  # Default temperature

        # Temperature thresholds
        # WHAT: Define safe temperature limits for each joint
        # WHY: Exceeding temperature limits can damage motors and other components
        self.warning_threshold = 60.0  # Celsius
        self.error_threshold = 70.0    # Celsius

        # Publishers and subscribers
        # WHAT: Create publisher for safety status and subscriber for temperature data
        # WHY: Communication with other nodes is essential for coordinated safety
        self.safety_status_publisher = self.create_publisher(String, '/safety_status', 10)
        self.temperature_subscriber = self.create_subscription(
            Temperature, '/joint_temperatures', self.temperature_callback, 10
        )
        self.power_reduction_publisher = self.create_publisher(Float64MultiArray, '/power_reduction', 10)

        # Timer for safety monitoring
        # WHAT: Create a timer to periodically check temperatures
        # WHY: Continuous monitoring is needed for safety
        self.monitor_timer = self.create_timer(1.0, self.safety_check)

        self.get_logger().info('Joint Temperature Monitor Agent initialized')

    def temperature_callback(self, msg: Temperature):
        """Process temperature messages from joints"""
        # WHAT: This method processes incoming temperature messages
        # WHY: Temperature data is needed for safety monitoring
        if hasattr(msg, 'name') and hasattr(msg, 'temperature'):
            joint_name = msg.name
            temperature = msg.temperature

            if joint_name in self.joint_temperatures:
                self.joint_temperatures[joint_name] = temperature
                self.get_logger().debug(f'Temperature for {joint_name}: {temperature}°C')

    def safety_check(self):
        """Check temperatures and update safety state"""
        # WHAT: This method checks all joint temperatures and updates the safety state
        # WHY: Regular safety checks ensure temperatures remain within safe limits
        max_temp = max(self.joint_temperatures.values())

        # Update safety state based on maximum temperature
        # WHAT: Determine the safety state based on the highest temperature
        # WHY: The overall safety state should reflect the most critical condition
        if max_temp > self.error_threshold:
            if self.safety_state != SafetyState.EMERGENCY_STOP:
                self.safety_state = SafetyState.EMERGENCY_STOP
                self.get_logger().error(f'EMERGENCY STOP: Temperature {max_temp}°C exceeds error threshold {self.error_threshold}°C')
                self.publish_emergency_stop()
        elif max_temp > self.warning_threshold:
            if self.safety_state != SafetyState.ERROR:
                self.safety_state = SafetyState.WARNING
                self.get_logger().warning(f'WARNING: Temperature {max_temp}°C exceeds warning threshold {self.warning_threshold}°C')
                self.publish_power_reduction()
        else:
            if self.safety_state != SafetyState.SAFE:
                self.safety_state = SafetyState.SAFE
                self.get_logger().info('All temperatures within safe limits')

        # Publish safety status
        self.publish_safety_status()

    def publish_safety_status(self):
        """Publish current safety status"""
        # WHAT: Publish the current safety status for other nodes to monitor
        # WHY: Other nodes may need to know the safety state for coordinated behavior
        status_msg = String()
        status_msg.data = f"Temperature Safety Status: {self.safety_state.value}, Max Temp: {max(self.joint_temperatures.values()):.2f}°C"
        self.safety_status_publisher.publish(status_msg)

    def publish_power_reduction(self):
        """Publish power reduction commands"""
        # WHAT: Send commands to reduce motor power when temperatures are high
        # WHY: Reducing power helps decrease heat generation and cool down joints
        reduction_msg = Float64MultiArray()
        # Reduce power to 50% when in warning state
        reduction_values = [0.5] * len(self.joint_names)
        reduction_msg.data = reduction_values
        self.power_reduction_publisher.publish(reduction_msg)

    def publish_emergency_stop(self):
        """Publish emergency stop commands"""
        # WHAT: Send emergency stop commands when temperatures are critically high
        # WHY: Immediate stop is needed to prevent damage when temperatures are critical
        stop_msg = Float64MultiArray()
        # Stop all joints when in emergency state
        stop_values = [0.0] * len(self.joint_names)
        stop_msg.data = stop_values
        self.power_reduction_publisher.publish(stop_msg)

def main(args=None):
    """Main function to initialize and run the temperature monitor agent"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    agent = JointTemperatureMonitorAgent()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(agent)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the agent when needed
        agent.get_logger().info('Shutting down temperature monitor agent')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`

### Exercise 6: State Machine Agent
**Difficulty**: Intermediate

Implement an agent using a state machine pattern that controls a humanoid robot's posture transitions (standing, sitting, lying down). Include proper state transition logic, safety checks, and error recovery.

```python
# State Machine Agent Implementation
# WHAT: This code creates an agent that controls humanoid robot posture transitions using a state machine
# WHY: To demonstrate state machine pattern for complex robot behaviors

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, List

class PostureState(Enum):
    """Enumeration of possible posture states"""
    STANDING = "standing"
    SITTING = "sitting"
    LYING_DOWN = "lying_down"
    TRANSITIONING = "transitioning"
    ERROR = "error"

class PostureControllerAgent(Node):
    def __init__(self):
        super().__init__('posture_controller_agent')

        # State management
        # WHAT: Track the current and target posture states
        # WHY: State tracking is essential for proper posture control
        self.current_state = PostureState.STANDING
        self.target_state = PostureState.STANDING
        self.previous_state = PostureState.STANDING
        self.state_start_time = self.get_clock().now()

        # Joint positions for each posture
        # WHAT: Define joint positions for different postures
        # WHY: Each posture requires specific joint angles for the robot
        self.posture_positions = {
            PostureState.STANDING: {
                'left_hip': 0.0, 'left_knee': -0.5, 'left_ankle': 0.5,
                'right_hip': 0.0, 'right_knee': -0.5, 'right_ankle': 0.5,
                'left_shoulder': 0.0, 'left_elbow': -0.5,
                'right_shoulder': 0.0, 'right_elbow': -0.5
            },
            PostureState.SITTING: {
                'left_hip': 1.0, 'left_knee': -1.5, 'left_ankle': 0.5,
                'right_hip': 1.0, 'right_knee': -1.5, 'right_ankle': 0.5,
                'left_shoulder': 0.2, 'left_elbow': -1.0,
                'right_shoulder': 0.2, 'right_elbow': -1.0
            },
            PostureState.LYING_DOWN: {
                'left_hip': 0.0, 'left_knee': 0.0, 'left_ankle': 0.0,
                'right_hip': 0.0, 'right_knee': 0.0, 'right_ankle': 0.0,
                'left_shoulder': 1.0, 'left_elbow': 0.0,
                'right_shoulder': 1.0, 'right_elbow': 0.0
            }
        }

        # Current joint positions
        self.current_positions = {name: 0.0 for name in self.posture_positions[PostureState.STANDING].keys()}
        self.desired_positions = {name: 0.0 for name in self.posture_positions[PostureState.STANDING].keys()}

        # Publishers and subscribers
        # WHAT: Create publishers for commands and status, subscribers for state feedback
        # WHY: Communication is needed for control and monitoring
        self.joint_command_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        self.posture_status_publisher = self.create_publisher(String, '/posture_status', 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Command subscriber
        self.command_subscriber = self.create_subscription(
            String, '/posture_command', self.command_callback, 10
        )

        # Control timer
        # WHAT: Create a timer to execute the state machine logic
        # WHY: Regular execution is needed for state transitions and control
        self.control_timer = self.create_timer(0.1, self.state_machine_step)

        self.get_logger().info('Posture Controller Agent initialized in STANDING state')

    def joint_state_callback(self, msg: JointState):
        """Update current joint positions"""
        # WHAT: Process incoming joint state messages to update current positions
        # WHY: Current positions are needed to determine transition progress
        for i, name in enumerate(msg.name):
            if name in self.current_positions and i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def command_callback(self, msg: String):
        """Handle posture change commands"""
        # WHAT: Process commands to change postures
        # WHY: External commands can trigger posture changes
        command = msg.data.lower().strip()

        if command == 'stand':
            self.request_posture_transition(PostureState.STANDING)
        elif command == 'sit':
            self.request_posture_transition(PostureState.SITTING)
        elif command == 'lie_down':
            self.request_posture_transition(PostureState.LYING_DOWN)
        else:
            self.get_logger().warning(f'Unknown posture command: {command}')

    def request_posture_transition(self, target_state: PostureState):
        """Request a transition to a new posture"""
        # WHAT: Request a transition to a new posture
        # WHY: Allow external requests to change the robot's posture
        if self.current_state != PostureState.TRANSITIONING:
            self.get_logger().info(f'Requesting transition from {self.current_state.value} to {target_state.value}')
            self.target_state = target_state
            self.previous_state = self.current_state
            self.current_state = PostureState.TRANSITIONING
            self.state_start_time = self.get_clock().now()

            # Set desired positions for the target posture
            for joint, position in self.posture_positions[target_state].items():
                self.desired_positions[joint] = position
        else:
            self.get_logger().warning('Cannot initiate transition: already transitioning')

    def state_machine_step(self):
        """Execute the state machine logic"""
        # WHAT: Execute the current state's logic
        # WHY: The state machine needs to run continuously to manage posture control
        if self.current_state == PostureState.TRANSITIONING:
            self.execute_transitioning_state()
        elif self.current_state == PostureState.ERROR:
            self.execute_error_state()
        else:
            # For stable states, just maintain position
            for joint, position in self.posture_positions[self.current_state].items():
                self.desired_positions[joint] = position

        # Publish commands and status
        self.publish_joint_commands()
        self.publish_posture_status()

    def execute_transitioning_state(self):
        """Execute the transitioning state logic"""
        # WHAT: Manage the transition from current to target posture
        # WHY: Transitions need to be smooth and controlled to avoid jerky movements
        elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

        # Calculate transition progress (simplified linear interpolation)
        # In a real implementation, this would use smoother interpolation
        progress = min(elapsed_time / 3.0, 1.0)  # 3 seconds for transition

        # Interpolate between current and target positions
        for joint in self.desired_positions.keys():
            current_pos = self.current_positions[joint]
            target_pos = self.posture_positions[self.target_state][joint]
            interpolated_pos = current_pos + (target_pos - current_pos) * progress
            self.desired_positions[joint] = interpolated_pos

        # Check if transition is complete
        if progress >= 1.0:
            self.current_state = self.target_state
            self.get_logger().info(f'Completed transition to {self.current_state.value}')

            # Set final positions
            for joint, position in self.posture_positions[self.current_state].items():
                self.desired_positions[joint] = position

    def execute_error_state(self):
        """Execute the error state logic"""
        # WHAT: Handle error conditions in the posture controller
        # WHY: Errors need to be managed to maintain robot safety
        self.get_logger().warning('Posture controller in ERROR state - maintaining current position')
        # In error state, maintain current position or move to safe position

    def publish_joint_commands(self):
        """Publish joint position commands"""
        # WHAT: Send desired joint positions to the robot
        # WHY: Commands must be sent to actuators for posture control
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = list(self.desired_positions.keys())
        cmd_msg.position = list(self.desired_positions.values())
        self.joint_command_publisher.publish(cmd_msg)

    def publish_posture_status(self):
        """Publish current posture status"""
        # WHAT: Publish the current posture state for monitoring
        # WHY: Other nodes may need to know the robot's current posture
        status_msg = String()
        status_msg.data = f"Current: {self.current_state.value}, Target: {self.target_state.value}"
        self.posture_status_publisher.publish(status_msg)

def main(args=None):
    """Main function to initialize and run the posture controller agent"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    agent = PostureControllerAgent()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(agent)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the agent when needed
        agent.get_logger().info('Shutting down posture controller agent')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`, `geometry_msgs`

### Exercise 7: Multi-Agent Coordination
**Difficulty**: Advanced

Implement a coordinator agent that manages multiple specialized agents (balance, walking, arm control) for a humanoid robot. Include proper coordination protocols, conflict resolution, and graceful degradation when agents fail.

```python
# Multi-Agent Coordination Implementation
# WHAT: This code creates a coordinator agent that manages multiple specialized agents for humanoid control
# WHY: To demonstrate coordination protocols and conflict resolution between specialized agents

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time
from enum import Enum
from typing import Dict, List
import threading
from dataclasses import dataclass

@dataclass
class AgentInfo:
    """Information about a managed agent"""
    name: str
    status: str
    last_seen: Time
    capabilities: List[str]
    priority: int

class SystemState(Enum):
    """Overall system states"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

class AgentCoordinator(Node):
    def __init__(self):
        super().__init__('agent_coordinator')

        # System state management
        # WHAT: Track the overall system state
        # WHY: The coordinator needs to know the overall health of the system
        self.system_state = SystemState.OPERATIONAL
        self.last_update_time = self.get_clock().now()

        # Managed agents registry
        # WHAT: Keep track of all managed agents and their information
        # WHY: The coordinator needs to monitor all agents it manages
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_priorities = {
            'balance_controller': 1,  # Highest priority
            'walking_controller': 2,
            'arm_controller': 3,
            'head_controller': 4,
            'navigation_agent': 5   # Lowest priority
        }

        # Publishers and subscribers
        # WHAT: Create communication channels for coordination
        # WHY: Communication is needed for status updates and command distribution
        self.status_publisher = self.create_publisher(String, '/coordinator_status', 10)
        self.command_publisher = self.create_publisher(String, '/system_commands', 10)
        self.emergency_publisher = self.create_publisher(Bool, '/emergency_stop', 10)

        # Subscribers for agent status and system control
        self.agent_status_subscriber = self.create_subscription(
            String, '/agent_status', self.agent_status_callback, 10
        )
        self.system_control_subscriber = self.create_subscription(
            String, '/system_control', self.system_control_callback, 10
        )

        # Timer for coordination logic
        # WHAT: Create a timer to run coordination logic periodically
        # WHY: Regular coordination checks are needed to maintain system health
        self.coordination_timer = self.create_timer(0.5, self.coordination_step)

        # Initialize agent registry
        self.register_default_agents()

        self.get_logger().info('Agent Coordinator initialized')

    def register_default_agents(self):
        """Register known agents in the system"""
        # WHAT: Register the default agents that are expected in the system
        # WHY: The coordinator needs to know which agents to monitor
        default_agents = [
            'balance_controller',
            'walking_controller',
            'arm_controller',
            'head_controller',
            'navigation_agent'
        ]

        for agent_name in default_agents:
            priority = self.agent_priorities.get(agent_name, 10)
            self.agents[agent_name] = AgentInfo(
                name=agent_name,
                status='unknown',
                last_seen=self.get_clock().now(),
                capabilities=[],
                priority=priority
            )
            self.get_logger().info(f'Registered agent: {agent_name} (priority: {priority})')

    def agent_status_callback(self, msg: String):
        """Process agent status updates"""
        # WHAT: Process status updates from managed agents
        # WHY: The coordinator needs to know the current status of each agent
        try:
            # Parse status message (format: "agent_name: status: capabilities")
            parts = msg.data.split(': ')
            if len(parts) >= 2:
                agent_name = parts[0].strip()
                status = parts[1].strip()

                if agent_name in self.agents:
                    self.agents[agent_name].status = status
                    self.agents[agent_name].last_seen = self.get_clock().now()

                    # Extract capabilities if provided
                    if len(parts) > 2:
                        capabilities_str = parts[2].strip()
                        self.agents[agent_name].capabilities = [cap.strip() for cap in capabilities_str.split(',')]

                    self.get_logger().debug(f'Updated status for {agent_name}: {status}')
                else:
                    # Register new agent if not in registry
                    priority = self.agent_priorities.get(agent_name, 10)
                    self.agents[agent_name] = AgentInfo(
                        name=agent_name,
                        status=status,
                        last_seen=self.get_clock().now(),
                        capabilities=[],
                        priority=priority
                    )
                    self.get_logger().info(f'Registered new agent: {agent_name}')
        except Exception as e:
            self.get_logger().error(f'Error parsing agent status message: {msg.data}, Error: {str(e)}')

    def system_control_callback(self, msg: String):
        """Process system control commands"""
        # WHAT: Process commands sent to the coordination system
        # WHY: External commands may need to affect multiple agents
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received system command: {command}')

        if command == 'emergency_stop':
            self.emergency_procedure()
        elif command.startswith('activate_'):
            agent_name = command.replace('activate_', '')
            self.activate_agent(agent_name)
        elif command.startswith('deactivate_'):
            agent_name = command.replace('deactivate_', '')
            self.deactivate_agent(agent_name)
        else:
            self.get_logger().warning(f'Unknown system command: {command}')

    def coordination_step(self):
        """Main coordination logic step"""
        # WHAT: Execute the main coordination logic
        # WHY: Regular coordination checks ensure system health and proper agent operation
        current_time = self.get_clock().now()
        timeout_threshold = rclpy.time.Duration(seconds=5.0)

        # Check for timed out agents
        # WHAT: Detect agents that haven't reported status recently
        # WHY: Unresponsive agents may need intervention or replacement
        for agent_name, agent_info in self.agents.items():
            time_since_seen = current_time - agent_info.last_seen
            if time_since_seen > timeout_threshold:
                if agent_info.status != 'error':
                    self.get_logger().warning(f'Agent {agent_name} appears to be unresponsive')
                    agent_info.status = 'timeout'
                    self.system_state = SystemState.DEGRADED

                    # Trigger recovery procedure for the agent
                    self.trigger_agent_recovery(agent_name)

        # Assess overall system health
        operational_count = sum(1 for info in self.agents.values() if info.status == 'operational')
        total_count = len(self.agents)

        if operational_count == 0:
            self.system_state = SystemState.EMERGENCY
        elif operational_count < total_count * 0.5:  # Less than 50% operational
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.OPERATIONAL

        # Publish coordinator status
        status_msg = String()
        status_msg.data = f"Coordinator: {self.system_state.value}, {operational_count}/{total_count} agents operational"
        self.status_publisher.publish(status_msg)

        self.last_update_time = current_time

    def trigger_agent_recovery(self, agent_name: str):
        """Trigger recovery procedure for a problematic agent"""
        # WHAT: Attempt to recover an agent that is not responding
        # WHY: Automatic recovery helps maintain system stability
        self.get_logger().info(f'Triggering recovery for agent: {agent_name}')

        # Send recovery command to the agent
        recovery_cmd = String()
        recovery_cmd.data = f'{agent_name}: recovery'
        self.command_publisher.publish(recovery_cmd)

    def emergency_procedure(self):
        """Execute emergency stop procedure"""
        # WHAT: Execute the emergency stop procedure for all agents
        # WHY: Safety is paramount and requires immediate action when needed
        self.get_logger().warning('Executing emergency stop procedure')

        # Publish emergency stop to all agents
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_publisher.publish(emergency_msg)

        # Send emergency commands to each agent
        for agent_name in self.agents.keys():
            emergency_cmd = String()
            emergency_cmd.data = f'{agent_name}: emergency_stop'
            self.command_publisher.publish(emergency_cmd)

        self.system_state = SystemState.EMERGENCY

    def activate_agent(self, agent_name: str):
        """Activate a specific agent"""
        # WHAT: Send activation command to a specific agent
        # WHY: Agents may need to be selectively activated or deactivated
        if agent_name in self.agents:
            activation_cmd = String()
            activation_cmd.data = f'{agent_name}: activate'
            self.command_publisher.publish(activation_cmd)
            self.get_logger().info(f'Sent activation command to {agent_name}')
        else:
            self.get_logger().warning(f'Attempted to activate unknown agent: {agent_name}')

    def deactivate_agent(self, agent_name: str):
        """Deactivate a specific agent"""
        # WHAT: Send deactivation command to a specific agent
        # WHY: Agents may need to be selectively activated or deactivated
        if agent_name in self.agents:
            deactivation_cmd = String()
            deactivation_cmd.data = f'{agent_name}: deactivate'
            self.command_publisher.publish(deactivation_cmd)
            self.get_logger().info(f'Sent deactivation command to {agent_name}')
        else:
            self.get_logger().warning(f'Attempted to deactivate unknown agent: {agent_name}')

def main(args=None):
    """Main function to initialize and run the agent coordinator"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    coordinator = AgentCoordinator()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing all callbacks
        # WHY: Without spinning, the node wouldn't execute its timer and subscriber callbacks
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the coordinator when needed
        coordinator.get_logger().info('Shutting down agent coordinator')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `std_msgs`, `geometry_msgs`

## Summary

These exercises cover the implementation of Python agents for humanoid robot control, from basic monitoring agents to complex state machines and multi-agent coordination systems. Each exercise builds on the concepts introduced in the chapter, providing practical implementations with proper error handling and safety considerations.