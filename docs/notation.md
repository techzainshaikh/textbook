---
sidebar_position: 3
title: Notation
description: Mathematical and technical notation used in the textbook
keywords: [notation, mathematical notation, robotics notation, symbols]
---

# Notation

This document defines the mathematical and technical notation used throughout the Physical AI and Humanoid Robotics textbook.

## Mathematical Notation

### Vectors and Matrices
- **v** - Vector (bold lowercase)
- **R** - Matrix (bold uppercase)
- <sup>A</sup>**p**<sub>B</sub> - Vector **p** representing point B with respect to frame A
- <sup>A</sup>**R**<sub>B</sub> - Rotation matrix from frame B to frame A

### Transformations
- **T** - Homogeneous transformation matrix
- <sup>A</sup>**T**<sub>B</sub> - Transformation matrix from frame B to frame A
- SE(3) - Special Euclidean group (3D transformations)

### Time and Derivatives
- ẋ - First time derivative of x
- ẍ - Second time derivative of x
- x(t) - Value of x at time t

## ROS 2 Notation

### Topics and Services
- `/topic_name` - ROS 2 topic
- `/service_name` - ROS 2 service
- `package_name/MessageType` - Message type from a package

### Nodes and Parameters
- `node_name` - ROS 2 node name
- `~parameter_name` - Private parameter for a node

## Robotics Notation

### Kinematics
- θ<sub>i</sub> - Joint angle for joint i
- d<sub>i</sub> - Joint offset for prismatic joint i
- a<sub>i</sub> - Link length for link i
- α<sub>i</sub> - Link twist for link i

### Dynamics
- τ - Joint torques
- **M**(**q**) - Mass matrix
- **C**(**q**, **q̇**) - Coriolis and centrifugal forces matrix
- **g**(**q**) - Gravity forces vector

## Control Theory
- G(s) - Transfer function in Laplace domain
- K<sub>p</sub>, K<sub>i</sub>, K<sub>d</sub> - Proportional, integral, and derivative gains for PID controller
- **x** - State vector
- **u** - Control input vector
- **y** - Output vector

## AI and Machine Learning
- **W** - Weight matrix in neural networks
- σ(·) - Activation function
- ∇ - Gradient operator
- L - Loss function
- θ - Model parameters