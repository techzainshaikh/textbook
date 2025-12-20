---
title: ROS 2 Actions for VLA Systems
sidebar_position: 5
description: Implementing ROS 2 action servers and clients for complex task execution in Vision-Language-Action systems
keywords: [ROS 2, actions, task execution, VLA, humanoid robotics, action servers, action clients]
---

# Chapter 4: ROS 2 Actions for VLA Systems

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement ROS 2 action servers for complex robot task execution
- Create action clients that can interact with VLA system components
- Design custom action messages for Vision-Language-Action workflows
- Handle action feedback, goals, and results in complex robotic tasks
- Integrate ROS 2 actions with LLM planning and perception systems

## Prerequisites

Students should have:
- Understanding of ROS 2 fundamentals (covered in Module 1)
- Knowledge of ROS 2 services and topics (covered in Module 1)
- Familiarity with action-based task execution concepts
- Experience with Python programming for ROS 2
- Basic understanding of task planning (covered in Chapter 3)

## Core Concepts

ROS 2 actions provide a way to execute long-running tasks with feedback and status updates. Unlike services, which are synchronous, or topics, which are asynchronous, actions combine both approaches to handle complex tasks that may take significant time to complete.

### Action Architecture

**Action Components:**
- **Goal**: Request to initiate a long-running task with parameters
- **Feedback**: Continuous updates during task execution
- **Result**: Final outcome of the completed task
- **Status**: Current state of the action (active, succeeded, cancelled, aborted)

**Action Messages:**
- **Action Goal**: Defines the input parameters for the action
- **Action Result**: Defines the output parameters upon completion
- **Action Feedback**: Defines the ongoing status updates during execution

### Action Patterns in VLA Systems

**Navigation Actions**: Moving to specific locations with obstacle avoidance
**Manipulation Actions**: Grasping, lifting, and placing objects
**Perception Actions**: Object detection and scene understanding
**Composite Actions**: Multi-step tasks combining navigation and manipulation

## Implementation

Let's implement ROS 2 actions for Vision-Language-Action systems:

### Custom Action Message Definitions

First, let's define custom action messages for VLA tasks:

```bash
# In your ROS 2 package, create the action directory and files:
mkdir -p action/{Navigation,Manipulation,Perception,Composite}
```

**Navigation.action:**
```
# Goal: Navigate to a target location
geometry_msgs/PoseStamped target_pose
float32 max_speed
bool avoid_obstacles

---
# Result: Navigation outcome
bool success
string message
float32 execution_time
geometry_msgs/PoseStamped final_pose

---
# Feedback: Current navigation status
string status
float32 progress_percentage
geometry_msgs/PoseStamped current_pose
float32 distance_remaining
```

**Manipulation.action:**
```
# Goal: Manipulate an object
string object_name
geometry_msgs/PoseStamped target_pose
string operation  # "pick", "place", "grasp", "release"
float32 grip_force

---
# Result: Manipulation outcome
bool success
string message
float32 execution_time
geometry_msgs/PoseStamped final_pose

---
# Feedback: Current manipulation status
string status
float32 progress_percentage
string current_operation
geometry_msgs/PoseStamped current_pose
bool object_detected
bool object_grasped
```

**Perception.action:**
```
# Goal: Perform perception task
string task_type  # "detect_objects", "recognize_scene", "track_object"
string[] target_objects
geometry_msgs/Point search_center
float32 search_radius

---
# Result: Perception outcome
bool success
string message
float32 execution_time
object_recognition_msgs/RecognizedObjectArray objects
geometry_msgs/PoseArray poses

---
# Feedback: Current perception status
string status
float32 progress_percentage
int32 objects_detected
geometry_msgs/PoseArray candidate_poses
```

**Composite.action:**
```
# Goal: Execute composite task
VLAAction[] sub_tasks  # Array of sub-actions to execute
bool continue_on_failure

---
# Result: Composite task outcome
bool success
string message
float32 execution_time
ActionResult[] sub_results

---
# Feedback: Current composite status
string status
float32 progress_percentage
int32 current_task_index
string current_task_description
bool[] task_completed
```

### Action Server Implementation

```python
#!/usr/bin/env python3
# vla_action_servers.py

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import custom action messages (assuming they're in your package)
from your_vla_package.action import Navigation, Manipulation, Perception, Composite
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header
import time
import threading
from typing import Optional
import math

class NavigationActionServer(Node):
    """
    Action server for navigation tasks in VLA systems
    """

    def __init__(self):
        super().__init__('vla_navigation_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Navigation,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers for navigation
        self.nav_status_pub = self.create_publisher(
            Navigation.Feedback, 'navigation_status', 10
        )

        self.get_logger().info('Navigation action server initialized')

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info(f'Received navigation goal: {goal_request.target_pose}')

        # Validate goal
        if self._validate_navigation_goal(goal_request):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _validate_navigation_goal(self, goal_request) -> bool:
        """Validate the navigation goal"""
        # Check if target pose is valid
        target = goal_request.target_pose.pose
        if math.isnan(target.position.x) or math.isnan(target.position.y):
            return False

        # Check if target is within workspace bounds
        workspace_bounds = {
            'x': [-2.0, 2.0],
            'y': [-2.0, 2.0],
            'z': [0.0, 1.0]
        }

        if (not workspace_bounds['x'][0] <= target.position.x <= workspace_bounds['x'][1] or
            not workspace_bounds['y'][0] <= target.position.y <= workspace_bounds['y'][1] or
            not workspace_bounds['z'][0] <= target.position.z <= workspace_bounds['z'][1]):
            return False

        return True

    async def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        self.get_logger().info('Executing navigation goal')

        feedback_msg = Navigation.Feedback()
        result = Navigation.Result()

        target_pose = goal_handle.request.target_pose
        max_speed = goal_handle.request.max_speed
        avoid_obstacles = goal_handle.request.avoid_obstacles

        # Initialize feedback
        feedback_msg.status = "Initializing navigation"
        feedback_msg.progress_percentage = 0.0
        feedback_msg.current_pose = self._get_current_pose()
        feedback_msg.distance_remaining = self._calculate_distance(
            feedback_msg.current_pose, target_pose
        )

        start_time = time.time()

        try:
            # Simulate navigation execution
            current_pose = feedback_msg.current_pose
            total_distance = feedback_msg.distance_remaining
            traveled_distance = 0.0

            while traveled_distance < total_distance and not goal_handle.is_cancel_requested:
                # Update current pose (simulated movement)
                current_pose = self._update_current_pose(current_pose, target_pose, max_speed)

                # Calculate progress
                distance_remaining = self._calculate_distance(current_pose, target_pose)
                traveled_distance = total_distance - distance_remaining
                progress = min(100.0, (traveled_distance / total_distance) * 100.0) if total_distance > 0 else 100.0

                # Update feedback
                feedback_msg.status = "Navigating to target"
                feedback_msg.progress_percentage = progress
                feedback_msg.current_pose = current_pose
                feedback_msg.distance_remaining = distance_remaining

                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)

                # Sleep to simulate real navigation
                time.sleep(0.1)

                # Check for obstacles if avoidance is enabled
                if avoid_obstacles and self._detect_obstacle(current_pose, target_pose):
                    feedback_msg.status = "Avoiding obstacle"
                    # Simulate obstacle avoidance maneuver
                    time.sleep(0.5)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Goal canceled"
                return result

            # Check if we reached the target
            final_distance = self._calculate_distance(current_pose, target_pose)
            if final_distance < 0.1:  # 10cm tolerance
                goal_handle.succeed()
                result.success = True
                result.message = "Successfully reached target pose"
            else:
                goal_handle.abort()
                result.success = False
                result.message = f"Failed to reach target, distance: {final_distance:.2f}m"

            result.execution_time = time.time() - start_time
            result.final_pose = current_pose

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f"Execution failed: {str(e)}"

        return result

    def _get_current_pose(self) -> PoseStamped:
        """Get current robot pose (simulated)"""
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _calculate_distance(self, pose1: PoseStamped, pose2: PoseStamped) -> float:
        """Calculate Euclidean distance between two poses"""
        dx = pose2.pose.position.x - pose1.pose.position.x
        dy = pose2.pose.position.y - pose1.pose.position.y
        dz = pose2.pose.position.z - pose1.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _update_current_pose(self, current_pose: PoseStamped, target_pose: PoseStamped, speed: float) -> PoseStamped:
        """Update current pose based on movement toward target"""
        # Calculate direction vector
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        dz = target_pose.pose.position.z - current_pose.pose.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance > 0.01:  # Avoid division by zero
            # Move towards target at specified speed
            scale = min(speed * 0.1 / distance, 1.0)  # 0.1s time step
            new_pose = PoseStamped()
            new_pose.header = current_pose.header
            new_pose.pose.position.x = current_pose.pose.position.x + dx * scale
            new_pose.pose.position.y = current_pose.pose.position.y + dy * scale
            new_pose.pose.position.z = current_pose.pose.position.z + dz * scale
            new_pose.pose.orientation = current_pose.pose.orientation
            return new_pose

        return current_pose

    def _detect_obstacle(self, current_pose: PoseStamped, target_pose: PoseStamped) -> bool:
        """Detect obstacles in the path (simulated)"""
        # Simple simulation: 20% chance of obstacle detection
        import random
        return random.random() < 0.2

class ManipulationActionServer(Node):
    """
    Action server for manipulation tasks in VLA systems
    """

    def __init__(self):
        super().__init__('vla_manipulation_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Manipulation,
            'manipulate_object',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Manipulation action server initialized')

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info(f'Received manipulation goal: {goal_request.operation}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the manipulation goal"""
        self.get_logger().info(f'Executing {goal_handle.request.operation} operation')

        feedback_msg = Manipulation.Feedback()
        result = Manipulation.Result()

        object_name = goal_handle.request.object_name
        target_pose = goal_handle.request.target_pose
        operation = goal_handle.request.operation
        grip_force = goal_handle.request.grip_force

        # Initialize feedback
        feedback_msg.status = f"Starting {operation} operation"
        feedback_msg.progress_percentage = 0.0
        feedback_msg.current_operation = operation
        feedback_msg.current_pose = target_pose
        feedback_msg.object_detected = True  # Simulated
        feedback_msg.object_grasped = False

        start_time = time.time()

        try:
            if operation == "pick":
                success = await self._execute_pick_operation(
                    goal_handle, feedback_msg, object_name, target_pose, grip_force
                )
            elif operation == "place":
                success = await self._execute_place_operation(
                    goal_handle, feedback_msg, target_pose
                )
            elif operation == "grasp":
                success = await self._execute_grasp_operation(
                    goal_handle, feedback_msg, target_pose, grip_force
                )
            elif operation == "release":
                success = await self._execute_release_operation(
                    goal_handle, feedback_msg
                )
            else:
                success = False
                result.message = f"Unknown operation: {operation}"

            if success:
                goal_handle.succeed()
                result.success = True
                result.message = f"Successfully completed {operation} operation"
            else:
                goal_handle.abort()
                result.success = False
                result.message = f"Failed to complete {operation} operation"

            result.execution_time = time.time() - start_time
            result.final_pose = target_pose

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f"Execution failed: {str(e)}"

        return result

    async def _execute_pick_operation(self, goal_handle, feedback_msg, object_name, target_pose, grip_force):
        """Execute pick operation"""
        # Approach object
        feedback_msg.status = f"Approaching {object_name}"
        feedback_msg.progress_percentage = 25.0
        feedback_msg.current_operation = "approach"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(1.0)

        # Grasp object
        feedback_msg.status = f"Grasping {object_name}"
        feedback_msg.progress_percentage = 50.0
        feedback_msg.current_operation = "grasp"
        feedback_msg.object_grasped = True
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(1.0)

        # Lift object
        feedback_msg.status = f"Lifting {object_name}"
        feedback_msg.progress_percentage = 75.0
        feedback_msg.current_operation = "lift"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        # Verify grasp
        feedback_msg.status = f"Verifying grasp of {object_name}"
        feedback_msg.progress_percentage = 100.0
        feedback_msg.current_operation = "verify"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        return True

    async def _execute_place_operation(self, goal_handle, feedback_msg, target_pose):
        """Execute place operation"""
        # Approach target location
        feedback_msg.status = "Approaching placement location"
        feedback_msg.progress_percentage = 33.0
        feedback_msg.current_operation = "approach"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(1.0)

        # Lower object
        feedback_msg.status = "Lowering object to placement location"
        feedback_msg.progress_percentage = 66.0
        feedback_msg.current_operation = "lower"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        # Release object
        feedback_msg.status = "Releasing object"
        feedback_msg.progress_percentage = 100.0
        feedback_msg.current_operation = "release"
        feedback_msg.object_grasped = False
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        return True

    async def _execute_grasp_operation(self, goal_handle, feedback_msg, target_pose, grip_force):
        """Execute grasp operation"""
        feedback_msg.status = "Moving to grasp position"
        feedback_msg.progress_percentage = 50.0
        feedback_msg.current_operation = "position"
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(1.0)

        feedback_msg.status = "Closing gripper"
        feedback_msg.progress_percentage = 100.0
        feedback_msg.current_operation = "grasp"
        feedback_msg.object_grasped = True
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        return True

    async def _execute_release_operation(self, goal_handle, feedback_msg):
        """Execute release operation"""
        feedback_msg.status = "Opening gripper"
        feedback_msg.progress_percentage = 100.0
        feedback_msg.current_operation = "release"
        feedback_msg.object_grasped = False
        goal_handle.publish_feedback(feedback_msg)
        await self._simulate_action_duration(0.5)

        return True

    async def _simulate_action_duration(self, duration: float):
        """Simulate action execution time"""
        start = time.time()
        while time.time() - start < duration:
            await asyncio.sleep(0.01)

class PerceptionActionServer(Node):
    """
    Action server for perception tasks in VLA systems
    """

    def __init__(self):
        super().__init__('vla_perception_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Perception,
            'perform_perception',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Perception action server initialized')

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info(f'Received perception goal: {goal_request.task_type}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the perception goal"""
        self.get_logger().info('Executing perception goal')

        feedback_msg = Perception.Feedback()
        result = Perception.Result()

        task_type = goal_handle.request.task_type
        target_objects = goal_handle.request.target_objects
        search_center = goal_request.search_center
        search_radius = goal_request.search_radius

        # Initialize feedback
        feedback_msg.status = f"Starting {task_type} task"
        feedback_msg.progress_percentage = 0.0
        feedback_msg.objects_detected = 0

        start_time = time.time()

        try:
            if task_type == "detect_objects":
                result = await self._execute_detect_objects(
                    goal_handle, feedback_msg, target_objects, search_center, search_radius
                )
            elif task_type == "recognize_scene":
                result = await self._execute_recognize_scene(
                    goal_handle, feedback_msg
                )
            elif task_type == "track_object":
                result = await self._execute_track_object(
                    goal_handle, feedback_msg, target_objects[0] if target_objects else None
                )
            else:
                goal_handle.abort()
                result.success = False
                result.message = f"Unknown task type: {task_type}"
                return result

            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

            result.execution_time = time.time() - start_time

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f"Execution failed: {str(e)}"

        return result

    async def _execute_detect_objects(self, goal_handle, feedback_msg, target_objects, search_center, search_radius):
        """Execute object detection task"""
        import random

        # Simulate scanning process
        for i in range(10):  # Simulate 10 scan steps
            feedback_msg.status = f"Scanning environment ({i+1}/10)"
            feedback_msg.progress_percentage = (i + 1) * 10.0
            feedback_msg.objects_detected = random.randint(0, len(target_objects) + 2)

            goal_handle.publish_feedback(feedback_msg)
            await self._simulate_action_duration(0.2)

        # Create mock detection results
        result = Perception.Result()
        result.success = True
        result.message = f"Detected {feedback_msg.objects_detected} objects"

        # Create mock recognized objects
        # In a real implementation, this would populate with actual detection data
        result.objects = []  # Would be populated with recognized objects
        result.poses = []    # Would be populated with object poses

        return result

    async def _execute_recognize_scene(self, goal_handle, feedback_msg):
        """Execute scene recognition task"""
        # Simulate scene analysis
        for i in range(5):
            feedback_msg.status = f"Analyzing scene - Step {i+1}/5"
            feedback_msg.progress_percentage = (i + 1) * 20.0
            feedback_msg.objects_detected = i + 1  # Simulated increasing detection

            goal_handle.publish_feedback(feedback_msg)
            await self._simulate_action_duration(0.5)

        result = Perception.Result()
        result.success = True
        result.message = "Scene analysis completed"
        result.objects = []  # Would be populated with scene understanding
        result.poses = []    # Would be populated with spatial relationships

        return result

    async def _execute_track_object(self, goal_handle, feedback_msg, target_object):
        """Execute object tracking task"""
        if not target_object:
            result = Perception.Result()
            result.success = False
            result.message = "No target object specified for tracking"
            return result

        # Simulate continuous tracking
        for i in range(20):  # Simulate 20 tracking steps
            feedback_msg.status = f"Tracking {target_object} - Frame {i+1}/20"
            feedback_msg.progress_percentage = min(100.0, (i + 1) * 5.0)  # Up to 100%
            feedback_msg.objects_detected = 1  # One object being tracked

            goal_handle.publish_feedback(feedback_msg)
            await self._simulate_action_duration(0.1)

        result = Perception.Result()
        result.success = True
        result.message = f"Successfully tracked {target_object} for 2 seconds"
        result.objects = []  # Would contain tracking results
        result.poses = []    # Would contain trajectory information

        return result

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    nav_server = NavigationActionServer()
    manip_server = ManipulationActionServer()
    percep_server = PerceptionActionServer()

    # Use MultiThreadedExecutor to handle multiple action servers
    executor = MultiThreadedExecutor()
    executor.add_node(nav_server)
    executor.add_node(manip_server)
    executor.add_node(percep_server)

    try:
        print("VLA Action Servers starting...")
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down VLA Action Servers...")
    finally:
        nav_server.destroy_node()
        manip_server.destroy_node()
        percep_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

```python
#!/usr/bin/env python3
# vla_action_clients.py

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration

# Import custom action messages
from your_vla_package.action import Navigation, Manipulation, Perception, Composite
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Header
import time
import asyncio
from typing import Optional, List, Dict, Any

class VLAActionClient(Node):
    """
    Client for interacting with VLA action servers
    """

    def __init__(self):
        super().__init__('vla_action_client')

        # Create action clients
        self.nav_client = ActionClient(self, Navigation, 'navigate_to_pose')
        self.manip_client = ActionClient(self, Manipulation, 'manipulate_object')
        self.percep_client = ActionClient(self, Perception, 'perform_perception')

        # Wait for action servers to be available
        self.nav_client.wait_for_server(timeout_sec=5.0)
        self.manip_client.wait_for_server(timeout_sec=5.0)
        self.percep_client.wait_for_server(timeout_sec=5.0)

        self.get_logger().info('VLA Action Client initialized')

    def create_navigation_goal(self, target_pose: PoseStamped, max_speed: float = 0.5,
                              avoid_obstacles: bool = True) -> Navigation.Goal:
        """Create a navigation goal"""
        goal = Navigation.Goal()
        goal.target_pose = target_pose
        goal.max_speed = max_speed
        goal.avoid_obstacles = avoid_obstacles
        return goal

    def create_manipulation_goal(self, object_name: str, target_pose: PoseStamped,
                               operation: str, grip_force: float = 50.0) -> Manipulation.Goal:
        """Create a manipulation goal"""
        goal = Manipulation.Goal()
        goal.object_name = object_name
        goal.target_pose = target_pose
        goal.operation = operation
        goal.grip_force = grip_force
        return goal

    def create_perception_goal(self, task_type: str, target_objects: List[str] = None,
                             search_center: Point = None, search_radius: float = 1.0) -> Perception.Goal:
        """Create a perception goal"""
        goal = Perception.Goal()
        goal.task_type = task_type
        goal.target_objects = target_objects or []
        goal.search_center = search_center or Point(x=0.0, y=0.0, z=0.0)
        goal.search_radius = search_radius
        return goal

    async def navigate_to_pose(self, target_pose: PoseStamped, max_speed: float = 0.5,
                              avoid_obstacles: bool = True,
                              timeout: float = 30.0) -> Optional[Navigation.Result]:
        """Send navigation goal and wait for result"""
        goal = self.create_navigation_goal(target_pose, max_speed, avoid_obstacles)

        # Send goal
        goal_future = await self.nav_client.send_goal_async(goal)

        if not goal_future.accepted:
            self.get_logger().error('Navigation goal rejected')
            return None

        self.get_logger().info('Navigation goal accepted, waiting for result...')

        # Get result
        result_future = goal_future.get_result_async()

        # Wait for result with timeout
        try:
            result_response = await asyncio.wait_for(
                result_future,
                timeout=timeout
            )
            return result_response.result
        except asyncio.TimeoutError:
            self.get_logger().error(f'Navigation goal timed out after {timeout}s')
            # Cancel the goal
            cancel_future = goal_future.cancel_goal_async()
            try:
                await cancel_future
            except:
                pass
            return None

    async def manipulate_object(self, object_name: str, target_pose: PoseStamped,
                               operation: str, grip_force: float = 50.0,
                               timeout: float = 20.0) -> Optional[Manipulation.Result]:
        """Send manipulation goal and wait for result"""
        goal = self.create_manipulation_goal(object_name, target_pose, operation, grip_force)

        # Send goal
        goal_future = await self.manip_client.send_goal_async(goal)

        if not goal_future.accepted:
            self.get_logger().error('Manipulation goal rejected')
            return None

        self.get_logger().info('Manipulation goal accepted, waiting for result...')

        # Get result
        result_future = goal_future.get_result_async()

        # Wait for result with timeout
        try:
            result_response = await asyncio.wait_for(
                result_future,
                timeout=timeout
            )
            return result_response.result
        except asyncio.TimeoutError:
            self.get_logger().error(f'Manipulation goal timed out after {timeout}s')
            # Cancel the goal
            cancel_future = goal_future.cancel_goal_async()
            try:
                await cancel_future
            except:
                pass
            return None

    async def perform_perception(self, task_type: str, target_objects: List[str] = None,
                                search_center: Point = None, search_radius: float = 1.0,
                                timeout: float = 15.0) -> Optional[Perception.Result]:
        """Send perception goal and wait for result"""
        goal = self.create_perception_goal(task_type, target_objects, search_center, search_radius)

        # Send goal
        goal_future = await self.percep_client.send_goal_async(goal)

        if not goal_future.accepted:
            self.get_logger().error('Perception goal rejected')
            return None

        self.get_logger().info('Perception goal accepted, waiting for result...')

        # Get result
        result_future = goal_future.get_result_async()

        # Wait for result with timeout
        try:
            result_response = await asyncio.wait_for(
                result_future,
                timeout=timeout
            )
            return result_response.result
        except asyncio.TimeoutError:
            self.get_logger().error(f'Perception goal timed out after {timeout}s')
            # Cancel the goal
            cancel_future = goal_future.cancel_goal_async()
            try:
                await cancel_future
            except:
                pass
            return None

    def subscribe_to_navigation_feedback(self, feedback_callback):
        """Subscribe to navigation feedback"""
        return self.nav_client._send_goal_future.add_feedback_callback(feedback_callback)

    def subscribe_to_manipulation_feedback(self, feedback_callback):
        """Subscribe to manipulation feedback"""
        return self.manip_client._send_goal_future.add_feedback_callback(feedback_callback)

    def subscribe_to_perception_feedback(self, feedback_callback):
        """Subscribe to perception feedback"""
        return self.percep_client._send_goal_future.add_feedback_callback(feedback_callback)

class VLACompositeActionServer(Node):
    """
    Action server for composite tasks that coordinate multiple actions
    """

    def __init__(self):
        super().__init__('vla_composite_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Composite,
            'execute_composite_task',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Create action clients for coordinating sub-actions
        self.nav_client = ActionClient(self, Navigation, 'navigate_to_pose')
        self.manip_client = ActionClient(self, Manipulation, 'manipulate_object')
        self.percep_client = ActionClient(self, Perception, 'perform_perception')

        # Wait for servers
        self.nav_client.wait_for_server(timeout_sec=5.0)
        self.manip_client.wait_for_server(timeout_sec=5.0)
        self.percep_client.wait_for_server(timeout_sec=5.0)

        self.get_logger().info('Composite action server initialized')

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info(f'Received composite task with {len(goal_request.sub_tasks)} sub-tasks')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request"""
        self.get_logger().info('Received cancel request for composite task')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the composite goal"""
        self.get_logger().info('Executing composite task')

        feedback_msg = Composite.Feedback()
        result = Composite.Result()

        sub_tasks = goal_handle.request.sub_tasks
        continue_on_failure = goal_handle.request.continue_on_failure

        # Initialize feedback
        feedback_msg.status = "Initializing composite task"
        feedback_msg.progress_percentage = 0.0
        feedback_msg.current_task_index = 0
        feedback_msg.current_task_description = "Starting composite execution"
        feedback_msg.task_completed = [False] * len(sub_tasks)

        start_time = time.time()

        try:
            results = []

            for i, sub_task in enumerate(sub_tasks):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "Composite task canceled"
                    result.execution_time = time.time() - start_time
                    return result

                # Update feedback for current task
                feedback_msg.current_task_index = i
                feedback_msg.current_task_description = f"Executing task {i+1}/{len(sub_tasks)}"
                feedback_msg.progress_percentage = (i / len(sub_tasks)) * 100.0

                goal_handle.publish_feedback(feedback_msg)

                # Execute the sub-task based on its type
                sub_result = await self._execute_sub_task(sub_task, goal_handle)
                results.append(sub_result)

                # Mark task as completed
                feedback_msg.task_completed[i] = True
                feedback_msg.status = f"Completed task {i+1}/{len(sub_tasks)}"
                feedback_msg.progress_percentage = ((i + 1) / len(sub_tasks)) * 100.0

                goal_handle.publish_feedback(feedback_msg)

                # Check if task failed and whether to continue
                if not sub_result.success and not continue_on_failure:
                    goal_handle.abort()
                    result.success = False
                    result.message = f"Sub-task {i} failed: {sub_result.message}"
                    result.execution_time = time.time() - start_time
                    result.sub_results = results
                    return result

            # All tasks completed successfully
            goal_handle.succeed()
            result.success = True
            result.message = f"All {len(sub_tasks)} sub-tasks completed successfully"
            result.execution_time = time.time() - start_time
            result.sub_results = results

        except Exception as e:
            goal_handle.abort()
            result.success = False
            result.message = f"Composite execution failed: {str(e)}"
            result.execution_time = time.time() - start_time

        return result

    async def _execute_sub_task(self, sub_task, goal_handle):
        """Execute a single sub-task"""
        # This is a simplified version - in practice, you'd have more sophisticated task routing
        if sub_task.type == "navigation":
            # Create and send navigation goal
            nav_goal = Navigation.Goal()
            nav_goal.target_pose = sub_task.target_pose
            nav_goal.max_speed = getattr(sub_task, 'max_speed', 0.5)
            nav_goal.avoid_obstacles = getattr(sub_task, 'avoid_obstacles', True)

            goal_future = await self.nav_client.send_goal_async(nav_goal)
            if goal_future.accepted:
                result_future = goal_future.get_result_async()
                try:
                    result_response = await result_future
                    return result_response.result
                except:
                    # Return failure result
                    nav_result = Navigation.Result()
                    nav_result.success = False
                    nav_result.message = "Navigation sub-task failed"
                    return nav_result
            else:
                nav_result = Navigation.Result()
                nav_result.success = False
                nav_result.message = "Navigation goal rejected"
                return nav_result

        elif sub_task.type == "manipulation":
            # Similar pattern for manipulation
            manip_goal = Manipulation.Goal()
            manip_goal.object_name = getattr(sub_task, 'object_name', '')
            manip_goal.target_pose = getattr(sub_task, 'target_pose', PoseStamped())
            manip_goal.operation = getattr(sub_task, 'operation', 'grasp')
            manip_goal.grip_force = getattr(sub_task, 'grip_force', 50.0)

            goal_future = await self.manip_client.send_goal_async(manip_goal)
            if goal_future.accepted:
                result_future = goal_future.get_result_async()
                try:
                    result_response = await result_future
                    return result_response.result
                except:
                    manip_result = Manipulation.Result()
                    manip_result.success = False
                    manip_result.message = "Manipulation sub-task failed"
                    return manip_result
            else:
                manip_result = Manipulation.Result()
                manip_result.success = False
                manip_result.message = "Manipulation goal rejected"
                return manip_result

        else:
            # Return failure for unknown task type
            result = type('GenericResult', (), {'success': False, 'message': f'Unknown task type: {sub_task.type}'})()
            return result

def create_vla_action_client() -> VLAActionClient:
    """Factory function to create a VLA action client"""
    return VLAActionClient()

def main():
    """Main function for VLA action client example"""
    rclpy.init()

    client = VLAActionClient()

    # Example usage
    print("VLA Action Client initialized. Ready to send action requests.")

    # Example: Create a target pose
    target_pose = PoseStamped()
    target_pose.header = Header()
    target_pose.header.frame_id = "map"
    target_pose.pose.position.x = 1.0
    target_pose.pose.position.y = 1.0
    target_pose.pose.position.z = 0.0
    target_pose.pose.orientation.w = 1.0

    print("Ready to send action requests. In a real implementation, this would connect to action servers.")

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Examples

### Example 1: VLA Task Coordination System

```python
#!/usr/bin/env python3
# vla_task_coordination.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
import asyncio
from typing import Dict, Any, List, Optional

class VLATaskCoordinationSystem(Node):
    """
    System that coordinates Vision-Language-Action tasks using ROS 2 actions
    """

    def __init__(self):
        super().__init__('vla_task_coordinator')

        # Create action clients for all VLA components
        self.action_client = VLAActionClient()

        # Task queue for managing multiple requests
        self.task_queue = asyncio.Queue()
        self.is_running = False

        self.get_logger().info('VLA Task Coordination System initialized')

    async def process_vla_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a VLA command that may involve navigation, manipulation, and perception
        """
        command_type = command.get('type', 'unknown')
        target_object = command.get('target_object', 'unknown')
        target_location = command.get('target_location', None)
        action = command.get('action', 'unknown')

        results = {}

        try:
            if command_type == 'navigation':
                # Navigate to target location
                if target_location:
                    pose_stamped = self._dict_to_pose_stamped(target_location)
                    nav_result = await self.action_client.navigate_to_pose(pose_stamped)
                    results['navigation'] = nav_result

            elif command_type == 'manipulation':
                # Perform manipulation task
                if target_object and target_location:
                    pose_stamped = self._dict_to_pose_stamped(target_location)
                    manip_result = await self.action_client.manipulate_object(
                        target_object, pose_stamped, action
                    )
                    results['manipulation'] = manip_result
                else:
                    # First detect the object
                    percep_result = await self.action_client.perform_perception(
                        'detect_objects', [target_object]
                    )
                    results['perception'] = percep_result

                    if percep_result and percep_result.success:
                        # Navigate to object
                        if percep_result.poses:
                            nav_result = await self.action_client.navigate_to_pose(
                                percep_result.poses[0]
                            )
                            results['navigation'] = nav_result

                            # Perform manipulation
                            manip_result = await self.action_client.manipulate_object(
                                target_object, percep_result.poses[0], action
                            )
                            results['manipulation'] = manip_result

            elif command_type == 'composite':
                # Execute a sequence of actions
                tasks = command.get('tasks', [])
                for task in tasks:
                    task_result = await self._execute_single_task(task)
                    results[f"task_{task.get('type', 'unknown')}"] = task_result

        except Exception as e:
            self.get_logger().error(f'Error processing VLA command: {e}')
            return {'success': False, 'error': str(e)}

        return {
            'success': all(res.success for res in results.values() if hasattr(res, 'success')),
            'results': results,
            'command': command
        }

    def _dict_to_pose_stamped(self, pose_dict: Dict[str, float]) -> PoseStamped:
        """Convert dictionary to PoseStamped message"""
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = pose_dict.get('frame_id', 'map')
        pose.pose.position.x = pose_dict.get('x', 0.0)
        pose.pose.position.y = pose_dict.get('y', 0.0)
        pose.pose.position.z = pose_dict.get('z', 0.0)

        # Default orientation (identity quaternion)
        pose.pose.orientation.w = 1.0
        return pose

    async def _execute_single_task(self, task: Dict[str, Any]) -> Any:
        """Execute a single task based on its type"""
        task_type = task.get('type', 'unknown')

        if task_type == 'navigation':
            target_pose = self._dict_to_pose_stamped(task.get('target_pose', {}))
            return await self.action_client.navigate_to_pose(target_pose)
        elif task_type == 'manipulation':
            target_pose = self._dict_to_pose_stamped(task.get('target_pose', {}))
            return await self.action_client.manipulate_object(
                task.get('object_name', 'unknown'),
                target_pose,
                task.get('operation', 'grasp')
            )
        elif task_type == 'perception':
            return await self.action_client.perform_perception(
                task.get('task_type', 'detect_objects'),
                task.get('target_objects', [])
            )
        else:
            # Return a generic failure result
            result = type('GenericResult', (), {'success': False, 'message': f'Unknown task type: {task_type}'})()
            return result

    async def run_task_scheduler(self):
        """Run the task scheduler to process queued tasks"""
        self.is_running = True

        while self.is_running:
            try:
                # Get next task from queue
                task = await self.task_queue.get()

                if task is None:  # Sentinel to stop the scheduler
                    break

                # Process the task
                result = await self.process_vla_command(task)

                # Log the result
                self.get_logger().info(f'Task completed: {result["success"]}')

                # Mark task as done
                self.task_queue.task_done()

            except Exception as e:
                self.get_logger().error(f'Error in task scheduler: {e}')
                continue

    def add_task(self, command: Dict[str, Any]):
        """Add a task to the queue"""
        self.task_queue.put_nowait(command)

    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.is_running = False
        # Add sentinel to stop the scheduler
        self.task_queue.put_nowait(None)

async def main():
    """Main function to demonstrate VLA task coordination"""
    rclpy.init()

    coordinator = VLATaskCoordinationSystem()

    # Example commands
    commands = [
        {
            'type': 'navigation',
            'target_location': {'x': 1.0, 'y': 1.0, 'z': 0.0, 'frame_id': 'map'}
        },
        {
            'type': 'manipulation',
            'target_object': 'red_cube',
            'action': 'pick',
            'target_location': {'x': 0.5, 'y': 0.5, 'z': 0.0}
        },
        {
            'type': 'composite',
            'tasks': [
                {
                    'type': 'perception',
                    'task_type': 'detect_objects',
                    'target_objects': ['blue_sphere']
                },
                {
                    'type': 'navigation',
                    'target_pose': {'x': 0.8, 'y': 0.8, 'z': 0.0}
                },
                {
                    'type': 'manipulation',
                    'object_name': 'blue_sphere',
                    'operation': 'grasp',
                    'target_pose': {'x': 0.8, 'y': 0.8, 'z': 0.1}
                }
            ]
        }
    ]

    # Add tasks to queue
    for cmd in commands:
        coordinator.add_task(cmd)

    # Run scheduler for a few seconds
    import threading
    scheduler_thread = threading.Thread(target=lambda: rclpy.spin(coordinator))
    scheduler_thread.start()

    try:
        await coordinator.run_task_scheduler()
    except KeyboardInterrupt:
        print("Shutting down VLA Task Coordination System...")
    finally:
        coordinator.stop_scheduler()
        coordinator.destroy_node()
        rclpy.shutdown()
        scheduler_thread.join()

if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2: Integration with LLM Planning System

```python
#!/usr/bin/env python3
# vla_llm_integration.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from typing import Dict, Any, List
import json

class VLALLMIntegration(Node):
    """
    Integration layer between LLM planning and ROS 2 action execution
    """

    def __init__(self):
        super().__init__('vla_llm_integration')

        # Action clients
        self.action_client = VLAActionClient()

        # Mapping from LLM actions to ROS 2 actions
        self.action_mapping = {
            'navigate_to': self._execute_navigation,
            'pick_object': self._execute_manipulation,
            'place_object': self._execute_manipulation,
            'grasp': self._execute_manipulation,
            'release': self._execute_manipulation,
            'inspect': self._execute_perception,
            'detect_object': self._execute_perception,
            'approach': self._execute_navigation
        }

        self.get_logger().info('VLA-LLM Integration initialized')

    async def execute_llm_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a plan generated by an LLM"""
        results = []
        success = True
        total_time = 0.0

        for i, step in enumerate(plan):
            action_name = step.get('action', '')
            parameters = step.get('parameters', {})

            self.get_logger().info(f'Executing step {i+1}/{len(plan)}: {action_name}')

            # Map LLM action to ROS 2 action
            if action_name in self.action_mapping:
                try:
                    start_time = self.get_clock().now().nanoseconds / 1e9
                    result = await self.action_mapping[action_name](parameters)
                    execution_time = (self.get_clock().now().nanoseconds / 1e9) - start_time

                    step_result = {
                        'step': i,
                        'action': action_name,
                        'parameters': parameters,
                        'result': result,
                        'execution_time': execution_time,
                        'success': result.success if hasattr(result, 'success') else True
                    }

                    results.append(step_result)
                    total_time += execution_time

                    if not step_result['success']:
                        success = False
                        self.get_logger().error(f'Step {i} failed: {action_name}')

                except Exception as e:
                    self.get_logger().error(f'Error executing step {i}: {e}')
                    step_result = {
                        'step': i,
                        'action': action_name,
                        'parameters': parameters,
                        'result': None,
                        'execution_time': 0.0,
                        'success': False,
                        'error': str(e)
                    }
                    results.append(step_result)
                    success = False
            else:
                self.get_logger().error(f'Unknown action: {action_name}')
                step_result = {
                    'step': i,
                    'action': action_name,
                    'parameters': parameters,
                    'result': None,
                    'execution_time': 0.0,
                    'success': False,
                    'error': f'Unknown action: {action_name}'
                }
                results.append(step_result)
                success = False

        return {
            'success': success,
            'results': results,
            'total_execution_time': total_time,
            'steps_completed': len([r for r in results if r['success']])
        }

    async def _execute_navigation(self, parameters: Dict[str, Any]):
        """Execute navigation action from LLM plan"""
        target_pose_dict = parameters.get('target_pose', parameters.get('location', {}))
        max_speed = parameters.get('max_speed', 0.5)
        avoid_obstacles = parameters.get('avoid_obstacles', True)

        target_pose = self._dict_to_pose_stamped(target_pose_dict)
        return await self.action_client.navigate_to_pose(target_pose, max_speed, avoid_obstacles)

    async def _execute_manipulation(self, parameters: Dict[str, Any]):
        """Execute manipulation action from LLM plan"""
        object_name = parameters.get('object', parameters.get('target_object', 'unknown'))
        target_pose_dict = parameters.get('target_pose', parameters.get('location', {}))
        operation = parameters.get('operation', parameters.get('action', 'grasp'))
        grip_force = parameters.get('grip_force', 50.0)

        target_pose = self._dict_to_pose_stamped(target_pose_dict)
        return await self.action_client.manipulate_object(object_name, target_pose, operation, grip_force)

    async def _execute_perception(self, parameters: Dict[str, Any]):
        """Execute perception action from LLM plan"""
        task_type = parameters.get('task_type', parameters.get('action', 'detect_objects'))
        target_objects = parameters.get('target_objects', parameters.get('objects', []))
        search_center = parameters.get('search_center', {'x': 0.0, 'y': 0.0, 'z': 0.0})
        search_radius = parameters.get('search_radius', 1.0)

        center_point = Point()
        center_point.x = search_center.get('x', 0.0)
        center_point.y = search_center.get('y', 0.0)
        center_point.z = search_center.get('z', 0.0)

        return await self.action_client.perform_perception(
            task_type, target_objects, center_point, search_radius
        )

    def _dict_to_pose_stamped(self, pose_dict: Dict[str, float]) -> PoseStamped:
        """Convert dictionary to PoseStamped message"""
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = pose_dict.get('frame_id', 'map')
        pose.pose.position.x = pose_dict.get('x', 0.0)
        pose.pose.position.y = pose_dict.get('y', 0.0)
        pose.pose.position.z = pose_dict.get('z', 0.0)

        # Handle orientation - if not provided, use identity quaternion
        if 'orientation' in pose_dict:
            orient = pose_dict['orientation']
            pose.pose.orientation.x = orient.get('x', 0.0)
            pose.pose.orientation.y = orient.get('y', 0.0)
            pose.pose.orientation.z = orient.get('z', 0.0)
            pose.pose.orientation.w = orient.get('w', 1.0)
        else:
            pose.pose.orientation.w = 1.0  # Default to identity

        return pose

def main():
    """Main function for VLA-LLM integration"""
    rclpy.init()

    integration = VLALLMIntegration()

    print("VLA-LLM Integration system initialized.")
    print("This system can execute plans generated by LLMs using ROS 2 actions.")
    print("In a real implementation, this would connect to both LLM planning and robot action servers.")

    integration.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

ROS 2 actions provide a powerful framework for executing complex, long-running tasks in Vision-Language-Action systems. Key benefits include:

- **Asynchronous Execution**: Long-running tasks don't block other operations
- **Progress Feedback**: Continuous updates during task execution
- **Cancellation Support**: Ability to interrupt running tasks
- **Result Reporting**: Comprehensive outcome information
- **Error Handling**: Robust error management and recovery

The integration of actions with LLM planning systems enables high-level task decomposition to be executed reliably on robotic platforms.

## Exercises

### Conceptual
1. Compare and contrast ROS 2 actions, services, and topics. When would you choose each communication pattern for different VLA system components?

### Logical
1. Design an error handling strategy for a composite action that coordinates navigation, manipulation, and perception. How would your system handle partial failures and ensure safe robot operation?

### Implementation
1. Implement a complete VLA action system with custom action messages, action servers for navigation/manipulation/perception, and a coordinating client that can execute complex tasks involving multiple sequential and parallel actions.