---
sidebar_position: 4
title: Chapter 3 - URDF Humanoid Modeling
description: Creating robot models for simulation and control using URDF
keywords: [urdf, robot modeling, humanoid, simulation, kinematics]
---

# Chapter 3 - URDF Humanoid Modeling

## Learning Objectives

By the end of this chapter, students will be able to:
1. Create complete URDF models for humanoid robots with proper kinematic chains
2. Define joints, links, and physical properties for realistic simulation
3. Implement visual and collision geometries for accurate physics simulation
4. Validate URDF models for correctness and completeness
5. Integrate URDF models with ROS 2 simulation environments

## Prerequisites

Before starting this chapter, students should have:
- Understanding of basic robotics kinematics and dynamics
- Knowledge of 3D coordinate systems and transformations
- Basic understanding of physics simulation concepts
- Completed previous chapters on ROS 2 communication patterns

## Core Concepts

### URDF (Unified Robot Description Format)

URDF is an XML-based format for representing a robot model. It defines the physical and kinematic properties of a robot, including:
- Links (rigid bodies)
- Joints (constraints between links)
- Visual and collision properties
- Inertial properties
- Transmission information

### Kinematic Chains in Humanoid Robots

Humanoid robots have complex kinematic structures with multiple chains:
- **Leg chains**: From pelvis to feet (typically 6-7 DOF per leg)
- **Arm chains**: From torso to hands (typically 7-8 DOF per arm)
- **Spine chain**: Connecting base to head
- **Head chain**: For gaze and interaction

### Degrees of Freedom (DOF)

The number of independent movements a joint or mechanism can perform. Humanoid robots typically have 30+ DOF to achieve human-like mobility.

### Forward and Inverse Kinematics

- **Forward Kinematics**: Calculate end-effector position from joint angles
- **Inverse Kinematics**: Calculate joint angles to achieve desired end-effector position

For the position vector **p** of an end-effector and joint angles **q**:
- Forward: **p** = FK(**q**)
- Inverse: **q** = IK(**p**)

## Implementation

### Basic URDF Structure

Here's a basic URDF file for a simple humanoid robot:

```xml
<!-- Example: Simple Humanoid Robot URDF -->
<!-- WHAT: This URDF defines a basic humanoid robot with simplified kinematic structure -->
<!-- WHY: To demonstrate the basic components and structure of a humanoid robot model -->

<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Fixed link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso link -->
  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
  </joint>

  <!-- Head link -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="yellow">
        <color rgba="1 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting torso to head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- Left arm (simplified) -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>
</robot>
```

### Complete Humanoid URDF Model

Here's a more complete URDF model for a humanoid robot with legs, arms, and head:

```xml
<!-- Example: Complete Humanoid Robot URDF -->
<!-- WHAT: This URDF defines a complete humanoid robot with legs, arms, and head -->
<!-- WHY: To demonstrate a realistic humanoid model with proper kinematic chains -->

<?xml version="1.0"?>
<robot name="complete_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Base link (pelvis) -->
  <link name="base_link">
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.2"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.6"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.6"/>
      </geometry>
    </collision>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2.0"/>
  </joint>

  <!-- Left Arm Chain -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_yaw" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.05 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="1.0" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.5" upper="0.0" effort="30" velocity="2.0"/>
  </joint>

  <link name="left_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
  </joint>

  <!-- Right Arm Chain (mirror of left) -->
  <link name="right_shoulder">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_yaw" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.05 -0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="2.0" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="2.5" effort="30" velocity="2.0"/>
  </joint>

  <link name="right_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wrist" type="revolute">
    <parent link="right_lower_arm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
  </joint>

  <!-- Left Leg Chain -->
  <link name="left_hip">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="0 0.08 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_pitch" type="revolute">
    <parent link="left_hip"/>
    <child link="left_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="2.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <!-- Right Leg Chain (mirror of left) -->
  <link name="right_hip">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0 -0.08 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_pitch" type="revolute">
    <parent link="right_hip"/>
    <child link="right_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="2.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

</robot>
```

### URDF Validation and Testing Python Script

Here's a Python script to validate URDF models:

```python
# Example: URDF Validation Script
# WHAT: This script validates URDF files for common errors and completeness
# WHY: To ensure URDF models are correct before using them in simulation or control

import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Tuple, Optional

class URDFValidator:
    def __init__(self, urdf_path: str):
        # WHAT: Initialize the validator with the path to the URDF file
        # WHY: The validator needs to know which file to validate
        self.urdf_path = urdf_path
        self.tree = None
        self.root = None
        self.errors = []
        self.warnings = []

    def load_urdf(self) -> bool:
        """Load the URDF file and parse it"""
        # WHAT: This method loads and parses the URDF XML file
        # WHY: The validator needs to access the XML structure to perform validation
        try:
            self.tree = ET.parse(self.urdf_path)
            self.root = self.tree.getroot()
            return True
        except ET.ParseError as e:
            # WHAT: This handles XML parsing errors
            # WHY: Invalid XML syntax makes the URDF unusable
            self.errors.append(f"XML Parse Error: {str(e)}")
            return False
        except FileNotFoundError:
            # WHAT: This handles cases where the file doesn't exist
            # WHY: Missing files are a critical error that must be reported
            self.errors.append(f"File not found: {self.urdf_path}")
            return False

    def validate_robot_element(self) -> bool:
        """Validate the root robot element"""
        # WHAT: This method validates the root <robot> element and its name attribute
        # WHY: The root element must be correct for the URDF to be valid
        if self.root.tag != 'robot':
            # WHAT: Check that the root element is indeed named 'robot'
            # WHY: URDF files must have 'robot' as their root element
            self.errors.append("Root element must be 'robot'")
            return False

        robot_name = self.root.get('name')
        if not robot_name:
            # WHAT: Check that the robot element has a name attribute
            # WHY: Robot names are required for identification and namespace management
            self.errors.append("Robot element must have a 'name' attribute")
            return False

        if not robot_name.strip():
            # WHAT: Check that the robot name is not empty
            # WHY: Empty names are not meaningful and can cause issues
            self.errors.append("Robot name cannot be empty")
            return False

        return True

    def validate_links(self) -> bool:
        """Validate all link elements"""
        # WHAT: This method validates all link elements in the URDF
        # WHY: Links are fundamental components that must be correctly defined
        links = self.root.findall('link')
        if not links:
            # WHAT: Check that at least one link exists
            # WHY: A robot without links is not valid
            self.errors.append("No links found in URDF")
            return False

        link_names = set()
        for link in links:
            name = link.get('name')
            if not name:
                # WHAT: Check that each link has a name attribute
                # WHY: Link names are required for referencing in joints
                self.errors.append("Link element missing 'name' attribute")
                continue

            if name in link_names:
                # WHAT: Check for duplicate link names
                # WHY: Duplicate names cause ambiguity in the kinematic chain
                self.errors.append(f"Duplicate link name: {name}")
                continue

            link_names.add(name)

            # Check for required elements within link
            inertial = link.find('inertial')
            if inertial is None:
                # WHAT: Check if the link has an inertial element
                # WHY: While not always required, inertial properties are important for simulation
                self.warnings.append(f"Link '{name}' missing inertial element")
            else:
                self.validate_inertial(inertial, name)

            visual = link.find('visual')
            if visual is None:
                # WHAT: Check if the link has a visual element
                # WHY: Visual elements are important for visualization
                self.warnings.append(f"Link '{name}' missing visual element")

            collision = link.find('collision')
            if collision is None:
                # WHAT: Check if the link has a collision element
                # WHY: Collision elements are important for physics simulation
                self.warnings.append(f"Link '{name}' missing collision element")

        return len(self.errors) == 0

    def validate_inertial(self, inertial_element, link_name: str):
        """Validate inertial element"""
        # WHAT: This method validates the inertial properties of a link
        # WHY: Inertial properties are critical for physics simulation
        mass = inertial_element.find('mass')
        if mass is None or mass.get('value') is None:
            # WHAT: Check if the mass element and value are present
            # WHY: Mass is required for physics simulation
            self.warnings.append(f"Link '{link_name}' inertial missing mass value")
            return

        try:
            mass_value = float(mass.get('value'))
            if mass_value <= 0:
                # WHAT: Check if the mass is positive
                # WHY: Negative or zero mass is physically invalid
                self.errors.append(f"Link '{link_name}' has non-positive mass: {mass_value}")
        except ValueError:
            # WHAT: Handle invalid mass values
            # WHY: Non-numeric mass values are invalid
            self.errors.append(f"Link '{link_name}' has invalid mass value: {mass.get('value')}")

        # Check inertia values
        # WHAT: Validate the inertia matrix values
        # WHY: Inertia values must be numeric and physically plausible
        inertia = inertial_element.find('inertia')
        if inertia is not None:
            for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                value = inertia.get(attr)
                if value is not None:
                    try:
                        float(value)
                    except ValueError:
                        # WHAT: Handle invalid inertia values
                        # WHY: Non-numeric inertia values are invalid
                        self.errors.append(f"Link '{link_name}' has invalid {attr} value: {value}")

    def validate_joints(self) -> bool:
        """Validate all joint elements"""
        # WHAT: This method validates all joint elements in the URDF
        # WHY: Joints define the kinematic relationships between links
        joints = self.root.findall('joint')
        if not joints:
            # WHAT: Check if joints exist (warning, not error)
            # WHY: Some simple models might not have joints
            self.warnings.append("No joints found in URDF")
            return True  # Not necessarily an error

        link_names = {link.get('name') for link in self.root.findall('link')}
        joint_names = set()

        for joint in joints:
            name = joint.get('name')
            if not name:
                # WHAT: Check if each joint has a name
                # WHY: Joint names are required for identification
                self.errors.append("Joint element missing 'name' attribute")
                continue

            if name in joint_names:
                # WHAT: Check for duplicate joint names
                # WHY: Duplicate names cause ambiguity
                self.errors.append(f"Duplicate joint name: {name}")
                continue

            joint_names.add(name)

            joint_type = joint.get('type')
            if not joint_type:
                # WHAT: Check if each joint has a type
                # WHY: Joint type determines how the joint behaves
                self.errors.append(f"Joint '{name}' missing 'type' attribute")
                continue

            # Check for valid joint types
            # WHAT: Validate that the joint type is one of the supported types
            # WHY: Only certain joint types are supported by URDF
            valid_types = ['revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar']
            if joint_type not in valid_types:
                self.errors.append(f"Joint '{name}' has invalid type: {joint_type}")

            # Check parent and child links exist
            # WHAT: Verify that referenced links exist in the model
            # WHY: Joints connect existing links; non-existent links are invalid
            parent = joint.find('parent')
            child = joint.find('child')

            if parent is None or parent.get('link') is None:
                self.errors.append(f"Joint '{name}' missing parent link")
            else:
                parent_link = parent.get('link')
                if parent_link not in link_names:
                    self.errors.append(f"Joint '{name}' references non-existent parent link: {parent_link}")

            if child is None or child.get('link') is None:
                self.errors.append(f"Joint '{name}' missing child link")
            else:
                child_link = child.get('link')
                if child_link not in link_names:
                    self.errors.append(f"Joint '{name}' references non-existent child link: {child_link}")

            # Check for continuous joints (they shouldn't have limits)
            # WHAT: Validate that continuous joints don't have limits
            # WHY: Continuous joints rotate infinitely and shouldn't have limits
            if joint_type == 'continuous':
                limit = joint.find('limit')
                if limit is not None:
                    self.warnings.append(f"Joint '{name}' is continuous but has limits defined")

            # Check for revolute/prismatic joints (they should have limits)
            # WHAT: Validate that revolute and prismatic joints have limits
            # WHY: These joints need limits to constrain their motion
            elif joint_type in ['revolute', 'prismatic']:
                limit = joint.find('limit')
                if limit is None:
                    self.warnings.append(f"Joint '{name}' is {joint_type} but has no limits defined")

        return len(self.errors) == 0

    def validate_geometry(self):
        """Validate geometry elements in visual and collision"""
        # WHAT: This method validates the geometry elements in visual and collision sections
        # WHY: Geometry elements must be properly defined for visualization and physics
        for link in self.root.findall('link'):
            for element_type in ['visual', 'collision']:
                for element in link.findall(element_type):
                    geometry = element.find('geometry')
                    if geometry is not None:
                        # Check for exactly one geometry type
                        # WHAT: Ensure only one geometry type is defined per element
                        # WHY: Multiple geometry types in one element are invalid
                        geometry_types = []
                        for geom_type in ['box', 'cylinder', 'sphere', 'mesh']:
                            if geometry.find(geom_type) is not None:
                                geometry_types.append(geom_type)

                        if len(geometry_types) != 1:
                            self.errors.append(
                                f"Link '{link.get('name')}' {element_type} has {len(geometry_types)} geometry types, expected exactly 1"
                            )
                        else:
                            geom_element = geometry.find(geometry_types[0])
                            if geometry_types[0] == 'mesh':
                                filename = geom_element.get('filename')
                                if filename and not os.path.isabs(filename):
                                    # Check if mesh file exists relative to URDF location
                                    # WHAT: Validate that referenced mesh files exist
                                    # WHY: Missing mesh files cause visualization and physics errors
                                    urdf_dir = os.path.dirname(self.urdf_path)
                                    mesh_path = os.path.join(urdf_dir, filename)
                                    if not os.path.exists(mesh_path):
                                        self.warnings.append(
                                            f"Mesh file not found: {mesh_path} for link '{link.get('name')}'"
                                        )

    def validate_kinematic_chain(self) -> bool:
        """Validate that the kinematic chain is valid (no loops, proper tree structure)"""
        # WHAT: This method validates the kinematic chain structure
        # WHY: URDF models must form a tree structure (no loops)
        joints = self.root.findall('joint')
        links = self.root.findall('link')

        # Create a graph of parent-child relationships
        # WHAT: Build a mapping of parent-child relationships
        # WHY: Needed to analyze the kinematic structure
        parent_to_child = {}
        child_to_parent = {}

        for joint in joints:
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')

            if parent_elem is not None and child_elem is not None:
                parent_name = parent_elem.get('link')
                child_name = child_elem.get('link')

                if parent_name and child_name:
                    if child_name in child_to_parent:
                        # WHAT: Check for multiple parents (invalid in tree structure)
                        # WHY: Each link should have at most one parent in a tree
                        self.errors.append(f"Link '{child_name}' has multiple parents")
                        return False

                    child_to_parent[child_name] = parent_name
                    if parent_name not in parent_to_child:
                        parent_to_child[parent_name] = []
                    parent_to_child[parent_name].append(child_name)

        # Find the root link (one without a parent)
        # WHAT: Identify the root of the kinematic tree
        # WHY: A valid URDF tree must have exactly one root
        all_link_names = {link.get('name') for link in links}
        root_links = all_link_names - set(child_to_parent.keys())

        if len(root_links) == 0:
            self.errors.append("No root link found - all links have parents (possible loop)")
            return False
        elif len(root_links) > 1:
            self.errors.append(f"Multiple root links found: {root_links}")
            return False

        # Check for loops by traversing the tree
        # WHAT: Detect cycles in the kinematic chain
        # WHY: Loops create invalid kinematic structures
        visited = set()
        def traverse(link_name):
            if link_name in visited:
                self.errors.append(f"Loop detected in kinematic chain at link: {link_name}")
                return False
            visited.add(link_name)

            if link_name in parent_to_child:
                for child in parent_to_child[link_name]:
                    if not traverse(child):
                        return False
            return True

        root = list(root_links)[0]
        if not traverse(root):
            return False

        return True

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations and return results"""
        # WHAT: This method runs all validation checks
        # WHY: Consolidates all validation into a single method
        self.errors = []
        self.warnings = []

        if not self.load_urdf():
            return False, self.errors, self.warnings

        # Run all validation checks
        robot_ok = self.validate_robot_element()
        links_ok = self.validate_links()
        joints_ok = self.validate_joints()
        self.validate_geometry()
        chain_ok = self.validate_kinematic_chain()

        success = robot_ok and links_ok and joints_ok and chain_ok
        return success, self.errors, self.warnings

def main(urdf_path: str = None):
    """Main function to validate a URDF file"""
    # WHAT: This is the main function that validates a URDF file
    # WHY: Provides an entry point for using the validator
    if urdf_path is None:
        # Use the example URDF file if none provided
        urdf_path = "simple_humanoid.urdf"  # This would be the path to your URDF file

    validator = URDFValidator(urdf_path)
    success, errors, warnings = validator.validate()

    print(f"URDF Validation Results for: {urdf_path}")
    print(f"Valid: {success}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if success and not warnings:
        print("\nURDF is valid with no warnings!")
    elif success:
        print(f"\nURDF is valid but has {len(warnings)} warnings.")

if __name__ == '__main__':
    # Example usage - you would call this with the path to your URDF file
    # main("path/to/your/robot.urdf")
    print("URDFValidator class defined. Use main(urdf_path) to validate a URDF file.")
```

**Dependencies**: Python standard library only (xml.etree.ElementTree, os, typing)

## Examples

### Example: URDF with Xacro for Parameterization

Xacro allows parameterization of URDF models:

```xml
<!-- Example: Humanoid Robot with Xacro -->
<!-- WHAT: This URDF uses Xacro to parameterize robot dimensions and properties -->
<!-- WHY: To enable easy customization of robot models with different dimensions -->

<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="param_humanoid">

  <!-- Define robot parameters -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_mass" value="15.0" />
  <xacro:property name="torso_mass" value="10.0" />
  <xacro:property name="arm_mass" value="1.5" />
  <xacro:property name="leg_mass" value="3.0" />

  <!-- Define dimensions -->
  <xacro:property name="base_size_x" value="0.2" />
  <xacro:property name="base_size_y" value="0.25" />
  <xacro:property name="base_size_z" value="0.2" />

  <xacro:property name="torso_height" value="0.6" />
  <xacro:property name="torso_width" value="0.2" />
  <xacro:property name="torso_depth" value="0.2" />

  <!-- Macro for creating a simple box link -->
  <xacro:macro name="box_link" params="name mass x y z *inertial *visual *collision">
    <link name="${name}">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="inertial"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="visual"/>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="collision"/>
      </collision>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <xacro:box_link name="base_link" mass="${base_mass}">
    <inertial>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
    <visual>
      <geometry>
        <box size="${base_size_x} ${base_size_y} ${base_size_z}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_size_x} ${base_size_y} ${base_size_z}"/>
      </geometry>
    </collision>
  </xacro:box_link>

  <!-- Torso -->
  <xacro:box_link name="torso" mass="${torso_mass}">
    <inertial>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_depth} ${torso_width} ${torso_height}"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_depth} ${torso_width} ${torso_height}"/>
      </geometry>
    </collision>
  </xacro:box_link>

  <!-- Joint connecting base to torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 ${base_size_z/2}" rpy="0 0 0"/>
  </joint>

  <!-- Macro for creating arm chain -->
  <xacro:macro name="arm_chain" params="side position_x position_y shoulder_yaw_limit shoulder_pitch_limit elbow_limit">
    <!-- Shoulder -->
    <link name="${side}_shoulder">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_shoulder_yaw" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${position_x} ${position_y} 0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-shoulder_yaw_limit}" upper="${shoulder_yaw_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}"/>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-shoulder_pitch_limit}" upper="${shoulder_pitch_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Lower arm -->
    <link name="${side}_lower_arm">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_elbow" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-elbow_limit}" upper="0.0" effort="30" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Create both arms using the macro -->
  <xacro:arm_chain side="left" position_x="0.05" position_y="0.1" shoulder_yaw_limit="1.57" shoulder_pitch_limit="2.0" elbow_limit="2.5"/>
  <xacro:arm_chain side="right" position_x="0.05" position_y="-0.1" shoulder_yaw_limit="1.57" shoulder_pitch_limit="2.0" elbow_limit="2.5"/>

  <!-- Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

## Summary

In this chapter, we've covered the fundamentals of URDF modeling for humanoid robots:
- **Basic URDF Structure**: Links, joints, and their properties
- **Kinematic Chains**: Creating proper parent-child relationships for arms and legs
- **Physical Properties**: Mass, inertia, and geometric definitions
- **Validation**: Techniques to ensure URDF models are correct
- **Xacro Parameterization**: Using macros and properties for reusable models

Proper URDF modeling is essential for humanoid robot simulation and control. A well-constructed URDF enables accurate physics simulation, visualization, and integration with ROS 2 tools.

## Exercises

### Conceptual
1. Explain the difference between visual and collision geometries in URDF. Why might you use different shapes for each?

### Logical
2. Design the URDF structure for a 28-DOF humanoid robot. Identify all links, joints, and their types. What are the minimum requirements for stable bipedal locomotion?

### Implementation
3. Create a Python script that generates a URDF file for a simplified humanoid robot based on user-specified parameters (height, weight, number of DOF). The script should ensure the resulting URDF is kinematically valid.