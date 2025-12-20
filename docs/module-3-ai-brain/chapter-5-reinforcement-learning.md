---
title: Reinforcement Learning for Robotics with Isaac Sim
sidebar_position: 6
description: Implementing reinforcement learning algorithms for humanoid robot control using Isaac Sim and NVIDIA Omniverse
keywords: [reinforcement learning, robotics, Isaac Sim, NVIDIA Omniverse, humanoid control, AI training]
---

# Chapter 5: Reinforcement Learning for Robotics with Isaac Sim

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamentals of reinforcement learning for robotics applications
- Implement RL algorithms for humanoid robot control using Isaac Sim
- Design reward functions for complex robotic tasks
- Train and validate RL policies in simulation before deployment
- Apply transfer learning techniques from simulation to real robots

## Prerequisites

Students should have:
- Understanding of basic machine learning concepts and algorithms
- Knowledge of humanoid robot kinematics and dynamics (covered in Module 1)
- Experience with Isaac Sim simulation environment (covered in Chapter 1)
- Basic Python programming skills for AI/ML frameworks
- Familiarity with control theory concepts

## Core Concepts

Reinforcement learning (RL) provides a framework for training robots to perform complex tasks through trial and error, using reward signals to guide behavior. In robotics, RL enables the learning of control policies that can handle complex, dynamic environments and tasks that are difficult to program explicitly.

### RL Framework for Robotics

**Agent-Environment Interaction:**
- **State Space (S)**: Robot's configuration, sensor readings, and environmental context
- **Action Space (A)**: Joint torques, velocities, or position commands
- **Reward Function (R)**: Scalar feedback for task success, efficiency, and safety
- **Policy (π)**: Mapping from states to actions that maximizes expected reward
- **Environment Dynamics**: Robot kinematics, dynamics, and environmental physics

**RL Algorithms for Robotics:**
- **Deep Q-Networks (DQN)**: For discrete action spaces and contact-rich tasks
- **Actor-Critic Methods**: For continuous control of joint torques and velocities
- **Proximal Policy Optimization (PPO)**: Stable policy gradient method for robotics
- **Soft Actor-Critic (SAC)**: Maximum entropy RL for exploration and robustness
- **Twin Delayed DDPG (TD3)**: Off-policy method for continuous control

### Isaac Sim RL Integration

Isaac Sim provides specialized RL capabilities:
- **Isaac Gym**: GPU-accelerated physics simulation for parallel training
- **RL Environments**: Pre-built environments for manipulation and locomotion
- **Observation Spaces**: Integrated sensor data, kinematics, and dynamics
- **Action Spaces**: Joint control interfaces with various command types
- **Reward Functions**: Configurable reward computation graphs

## Implementation

Let's implement reinforcement learning for humanoid robot control using Isaac Sim:

### Isaac Sim RL Environment Setup

```python
#!/usr/bin/env python3
# rl_environment.py

import numpy as np
import torch
import gym
from gym import spaces
from pxr import Usd, UsdGeom, Gf
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.torch.maths import torch_acos, torch_cross, torch_normalize
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Any

class HumanoidRLEnvironment:
    """
    Reinforcement learning environment for humanoid robot control using Isaac Sim
    """

    def __init__(self, world: World, robot_path: str = "/World/Franka/robot"):
        self.world = world
        self.robot = self.world.scene.get_articulation(robot_path)

        # RL environment parameters
        self.max_episode_length = 1000
        self.current_step = 0
        self.episode_reward = 0.0

        # Define action and observation spaces
        self.num_actions = len(self.robot.dof_names)
        self.num_observations = 14 + 14 + 3 + 3 + 3  # joint_pos + joint_vel + root_pos + root_rot + target_pos

        # Action space: joint position targets (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        # Observation space: joint positions, velocities, root pose, target position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32
        )

        # Robot state tracking
        self.initial_joint_positions = None
        self.target_position = np.array([0.5, 0.0, 0.0])  # Target position in world frame

        self.get_logger().info('Humanoid RL Environment initialized')

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Reset robot to initial configuration
        if self.initial_joint_positions is None:
            self.initial_joint_positions = self.robot.get_joint_positions()

        self.robot.set_joint_positions(self.initial_joint_positions)
        self.current_step = 0
        self.episode_reward = 0.0

        # Randomize target position for training diversity
        self.target_position = np.random.uniform(low=-0.5, high=0.5, size=3)
        self.target_position[2] = 0.0  # Keep target on ground plane

        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of the environment"""
        # Apply action to robot
        self.apply_action(action)

        # Step the physics simulation
        self.world.step(render=True)

        # Get current observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()
        self.episode_reward += reward

        # Check if episode is done
        done = self.current_step >= self.max_episode_length
        self.current_step += 1

        # Additional info for debugging
        info = {
            'episode_reward': self.episode_reward,
            'distance_to_target': np.linalg.norm(observation[28:31] - self.target_position)
        }

        return observation, reward, done, info

    def apply_action(self, action: np.ndarray):
        """Apply action to the robot"""
        # Convert normalized action to joint position targets
        joint_targets = action * 0.5  # Scale to reasonable range

        # Apply position control to robot
        self.robot.set_joint_position_targets(joint_targets)

    def get_observation(self) -> np.ndarray:
        """Get current observation from environment"""
        # Get robot state
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        root_position = self.robot.get_world_poses()[0][0].cpu().numpy()
        root_rotation = self.robot.get_world_poses()[1][0].cpu().numpy()

        # Create observation vector
        obs = np.concatenate([
            joint_positions,      # 14 joint positions
            joint_velocities,     # 14 joint velocities
            root_position,        # 3 root position
            root_rotation,        # 4 root rotation (quaternion)
            self.target_position  # 3 target position
        ])

        return obs

    def calculate_reward(self) -> float:
        """Calculate reward based on current state"""
        # Get current position
        current_pos = self.robot.get_world_poses()[0][0].cpu().numpy()

        # Distance to target reward
        distance_to_target = np.linalg.norm(current_pos - self.target_position)
        distance_reward = -distance_to_target  # Negative distance encourages approach

        # Velocity penalty to encourage smooth motion
        joint_velocities = self.robot.get_joint_velocities()
        velocity_penalty = -0.01 * np.sum(np.square(joint_velocities))

        # Energy efficiency reward (penalize high torques)
        joint_efforts = self.robot.get_measured_joint_efforts()
        energy_penalty = -0.001 * np.sum(np.abs(joint_efforts))

        # Safety reward (penalize joint limits)
        joint_positions = self.robot.get_joint_positions()
        joint_limits = self.robot.get_dof_properties()['upper'][:len(joint_positions)]
        safety_penalty = 0.0
        for i, (pos, limit) in enumerate(zip(joint_positions, joint_limits)):
            if abs(pos) > 0.9 * abs(limit):  # Within 10% of joint limit
                safety_penalty -= 0.1

        total_reward = distance_reward + velocity_penalty + energy_penalty + safety_penalty
        return total_reward

    def get_logger(self):
        """Get logger instance"""
        import logging
        return logging.getLogger(__name__)

class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization agent for humanoid control
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOAgent, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output actions in [-1, 1] range
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output value estimate
        )

    def forward(self, x):
        """Forward pass through both actor and critic"""
        action_mean = self.actor(x)
        value = self.critic(x)
        return action_mean, value

    def get_action(self, x):
        """Sample action from policy"""
        action_mean, value = self.forward(x)
        # In PPO, we typically add noise during training but not during inference
        return action_mean, value

class PPOTrainer:
    """
    PPO training loop for humanoid robot control
    """

    def __init__(self, agent: PPOAgent, lr: float = 3e-4, gamma: float = 0.99,
                 eps_clip: float = 0.2, k_epochs: int = 4):
        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def train_step(self, states, actions, rewards, logprobs, state_values, masks):
        """Perform one training step of PPO"""
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0

        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_reward = reward + self.gamma * running_reward * mask
            discounted_rewards.insert(0, running_reward)

        # Normalize discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Convert to tensors
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(state_values, dim=0)).detach()

        # Calculate advantages
        advantages = discounted_rewards - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values = self.agent.get_action(old_states)

            # Find the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, discounted_rewards.unsqueeze(1))

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def main():
    """Main training loop"""
    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Add robot to the stage
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find Isaac Sim assets. Please enable Isaac Sim Nucleus server.")
        return

    # Load robot (example with a generic humanoid)
    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Robots/Franka/franka_instanceable.usd",
        prim_path="/World/Robot"
    )

    # Create RL environment
    env = HumanoidRLEnvironment(world, "/World/Robot")

    # Create PPO agent
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(observation_dim, action_dim)
    trainer = PPOTrainer(agent)

    # Training parameters
    num_episodes = 1000
    max_timesteps = 1000

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get action from agent
            with torch.no_grad():
                action_mean, _ = agent.get_action(state_tensor)
                action = action_mean.cpu().numpy()[0]

            # Execute action in environment
            next_state, reward, done, info = env.step(action)

            # Update state
            state = next_state
            episode_reward += reward

            if done:
                break

        # Print episode statistics
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    print("Training completed!")

if __name__ == "__main__":
    main()
```

### Isaac Sim RL Training Pipeline

```python
#!/usr/bin/env python3
# rl_training_pipeline.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import pickle
from typing import List, Tuple, Dict, Any

class RLTrainingPipeline:
    """
    Complete RL training pipeline for humanoid robot control
    """

    def __init__(self, agent: nn.Module, env, log_dir: str = "./logs"):
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

        # Training parameters
        self.batch_size = 1024
        self.update_timestep = 2000  # Update policy every n timesteps
        self.max_episodes = 10000
        self.max_timesteps = 1000
        self.print_freq = 10
        self.save_freq = 100

        # Storage for training data
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

        # Training statistics
        self.running_reward = 0
        self.avg_length = 0
        self.time_step = 0

    def train(self):
        """Main training loop"""
        print(f"Starting RL training with {self.max_episodes} episodes...")

        for episode in range(1, self.max_episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0

            for t in range(self.max_timesteps):
                # Select action
                action, logprob, state_value = self.select_action(state)

                # Store data
                self.states.append(torch.FloatTensor(state))
                self.actions.append(torch.FloatTensor(action))
                self.logprobs.append(logprob)
                self.state_values.append(state_value)

                # Execute action
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                self.is_terminals.append(done)

                # Update statistics
                episode_reward += reward
                self.time_step += 1

                if done:
                    break

            self.running_reward += episode_reward
            self.avg_length += t

            # Update policy if enough data collected
            if self.time_step % self.update_timestep == 0:
                self.update()
                self.time_step = 0

            # Print average reward every n episodes
            if episode % self.print_freq == 0:
                avg_reward = self.running_reward / self.print_freq
                avg_length = int(self.avg_length / self.print_freq)

                print(f'Episode: {episode}, Average Reward: {avg_reward:.2f}, Average Length: {avg_length}')

                self.writer.add_scalar('Reward/Average', avg_reward, episode)
                self.writer.add_scalar('Episode/Length', avg_length, episode)

                self.running_reward = 0
                self.avg_length = 0

            # Save model every n episodes
            if episode % self.save_freq == 0:
                self.save_model(f"rl_model_episode_{episode}.pth")

        # Close tensorboard writer
        self.writer.close()
        print("Training completed!")

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, state_value = self.agent.get_action(state_tensor)

            # For PPO, we add noise during training
            action = torch.normal(action_mean, 0.1)  # Add exploration noise
            action = torch.clamp(action, -1, 1)  # Ensure action is within bounds

            # Calculate log probability
            action_logprob = torch.log(1 - torch.tanh(action)**2 + 1e-6)  # Log prob for tanh output

        return action.cpu().numpy()[0], action_logprob, state_value

    def update(self):
        """Update policy using collected data"""
        # Convert lists to tensors
        old_states = torch.stack(self.states).detach()
        old_actions = torch.stack(self.actions).detach()
        old_logprobs = torch.stack(self.logprobs).detach()
        old_state_values = torch.stack(self.state_values).detach()

        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (0.99 * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Calculate advantages
        advantages = rewards - old_state_values.detach()

        # Optimize policy
        for _ in range(4):  # PPO epochs
            # Evaluate old actions and values using current policy
            logprobs, state_values = self.agent.get_action(old_states)

            # Find the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages

            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * torch.mean(logprobs)

            # Perform backpropagation
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()

        # Clear stored data
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def save_model(self, path: str):
        """Save trained model"""
        torch.save(self.agent.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        self.agent.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

class CurriculumLearning:
    """
    Curriculum learning for progressive skill acquisition
    """

    def __init__(self, env, difficulty_levels: List[Dict[str, Any]]):
        self.env = env
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        self.success_threshold = 0.8  # 80% success rate to advance
        self.episode_count = 0
        self.success_count = 0

    def update_curriculum(self, episode_reward: float, target_reward: float) -> bool:
        """Update curriculum based on performance"""
        self.episode_count += 1
        if episode_reward >= target_reward * self.success_threshold:
            self.success_count += 1

        # Check if we should advance to next level
        if (self.episode_count >= 100 and
            self.success_count / self.episode_count >= self.success_threshold):

            if self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                self.episode_count = 0
                self.success_count = 0
                print(f"Advancing to difficulty level {self.current_level + 1}")

                # Update environment parameters for new level
                self.update_environment()
                return True

        return False

    def update_environment(self):
        """Update environment parameters for current difficulty level"""
        level_params = self.difficulty_levels[self.current_level]

        # Update environment based on current level
        # This would typically involve changing task complexity, obstacles, etc.
        print(f"Updated environment for difficulty level {self.current_level + 1}")

def create_humanoid_rl_environment():
    """Factory function to create a complete humanoid RL environment"""
    # This would integrate with Isaac Sim to create a complete environment
    # For this example, we'll return a placeholder that would connect to Isaac Sim
    pass

def main():
    """Main function to run the RL training pipeline"""
    # Initialize Isaac Sim environment (simplified for example)
    # In practice, this would connect to the actual Isaac Sim simulation

    print("Setting up RL training environment...")

    # Create agent (in practice, this would connect to the Isaac Sim environment)
    # observation_dim = 41  # Based on humanoid state space
    # action_dim = 14       # Based on humanoid DOF
    # agent = PPOAgent(observation_dim, action_dim)

    # Create training pipeline
    # pipeline = RLTrainingPipeline(agent, env)

    # Run training
    # pipeline.train()

    print("RL training pipeline setup complete!")
    print("Note: Full implementation requires Isaac Sim integration for actual simulation")

if __name__ == "__main__":
    main()
```

## Examples

### Example 1: Humanoid Locomotion Training with PPO

```python
#!/usr/bin/env python3
# humanoid_locomotion_ppo.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class HumanoidLocomotionEnv:
    """
    Specialized environment for humanoid locomotion training
    """

    def __init__(self):
        # Define locomotion-specific parameters
        self.target_velocity = 1.0  # m/s
        self.max_episode_length = 500
        self.current_step = 0

        # State space: [joint_pos, joint_vel, root_pos, root_vel, target_pos]
        self.state_dim = 14 + 14 + 3 + 3 + 3  # Example dimensions

        # Action space: joint position targets
        self.action_dim = 14  # Example joint DOF

        # Reward weights
        self.velocity_weight = 1.0
        self.energy_weight = -0.01
        self.balance_weight = 0.5
        self.forward_weight = 0.8

    def reset(self) -> np.ndarray:
        """Reset environment for locomotion task"""
        self.current_step = 0

        # Initialize robot in standing position
        state = self.get_state()
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step of locomotion task"""
        # Apply action to robot
        self.apply_action(action)

        # Get new state
        next_state = self.get_state()

        # Calculate locomotion-specific reward
        reward = self.calculate_locomotion_reward()

        # Check termination
        done = self.current_step >= self.max_episode_length
        self.current_step += 1

        info = {
            'velocity': self.get_forward_velocity(),
            'balance': self.get_balance_score()
        }

        return next_state, reward, done, info

    def calculate_locomotion_reward(self) -> float:
        """Calculate reward for locomotion task"""
        # Get current state
        state = self.get_state()

        # Forward velocity reward
        forward_vel = self.get_forward_velocity()
        velocity_reward = self.velocity_weight * abs(forward_vel - self.target_velocity)

        # Energy efficiency penalty
        joint_velocities = state[14:28]  # Assuming joint velocities are in this range
        energy_penalty = self.energy_weight * np.sum(np.square(joint_velocities))

        # Balance reward (keep torso upright)
        balance_reward = self.balance_weight * self.get_balance_score()

        # Forward progress reward
        forward_progress = self.forward_weight * forward_vel

        total_reward = velocity_reward + energy_penalty + balance_reward + forward_progress
        return total_reward

    def get_forward_velocity(self) -> float:
        """Get forward velocity of humanoid"""
        # This would interface with Isaac Sim to get actual velocity
        return 0.0  # Placeholder

    def get_balance_score(self) -> float:
        """Get balance score (0-1, 1 being perfectly balanced)"""
        # This would interface with Isaac Sim to get actual balance
        return 0.0  # Placeholder

    def get_state(self) -> np.ndarray:
        """Get current state from Isaac Sim"""
        # This would interface with Isaac Sim to get actual state
        return np.zeros(self.state_dim)  # Placeholder

    def apply_action(self, action: np.ndarray):
        """Apply action to Isaac Sim robot"""
        # This would interface with Isaac Sim to apply actions
        pass

class LocomotionPPOAgent(nn.Module):
    """
    Specialized PPO agent for humanoid locomotion
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(LocomotionPPOAgent, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor network
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.1)

        # Critic network
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        """Forward pass through networks"""
        features = self.feature_extractor(state)

        # Actor output
        action_mean = torch.tanh(self.actor_mean(features))  # Keep actions bounded

        # Critic output
        value = self.critic(features)

        return action_mean, self.actor_std, value

def train_locomotion():
    """Train locomotion policy"""
    env = HumanoidLocomotionEnv()
    agent = LocomotionPPOAgent(env.state_dim, env.action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    # Training loop would go here
    # This is a simplified example of the training process

    print("Locomotion training setup complete")

if __name__ == "__main__":
    train_locomotion()
```

### Example 2: RL Policy Validation and Transfer

```python
#!/usr/bin/env python3
# rl_policy_validation.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

class PolicyValidator:
    """
    Validate trained RL policies for safety and performance
    """

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.metrics = {
            'success_rate': [],
            'episode_length': [],
            'cumulative_reward': [],
            'safety_violations': [],
            'energy_efficiency': []
        }

    def validate_policy(self, num_episodes: int = 100) -> Dict[str, float]:
        """Validate policy across multiple episodes"""
        total_success = 0
        total_reward = 0
        total_safety_violations = 0
        total_energy = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            safety_violations = 0
            energy_consumed = 0
            done = False

            while not done:
                # Get action from trained policy
                with torch.no_grad():
                    action_mean, _, _ = self.agent.forward(torch.FloatTensor(state).unsqueeze(0))
                    action = action_mean.squeeze(0).numpy()

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Track metrics
                episode_reward += reward
                energy_consumed += self.calculate_energy(action)

                # Check for safety violations
                if self.check_safety_violation(next_state):
                    safety_violations += 1

                state = next_state

            # Update statistics
            if self.check_task_success(info):
                total_success += 1

            total_reward += episode_reward
            total_safety_violations += safety_violations
            total_energy += energy_consumed

        # Calculate final metrics
        results = {
            'success_rate': total_success / num_episodes,
            'average_reward': total_reward / num_episodes,
            'average_safety_violations': total_safety_violations / num_episodes,
            'average_energy_efficiency': total_energy / num_episodes
        }

        return results

    def calculate_energy(self, action: np.ndarray) -> float:
        """Calculate energy consumption based on action"""
        return np.sum(np.abs(action))  # Simplified energy calculation

    def check_safety_violation(self, state: np.ndarray) -> bool:
        """Check if state violates safety constraints"""
        # Example: check if joint angles are within safe limits
        joint_positions = state[:14]  # Assuming first 14 elements are joint positions
        joint_limits = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # Example limits

        return np.any(np.abs(joint_positions) > joint_limits)

    def check_task_success(self, info: Dict) -> bool:
        """Check if task was completed successfully"""
        # This would depend on the specific task
        # For example, reaching a target position
        return info.get('distance_to_target', float('inf')) < 0.1  # Within 10cm of target

    def visualize_validation_results(self, results: Dict[str, float]):
        """Visualize validation results"""
        metrics = list(results.keys())
        values = list(results.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        plt.title('RL Policy Validation Results')
        plt.ylabel('Value')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

class SimToRealTransfer:
    """
    Handle transfer of policies from simulation to real robots
    """

    def __init__(self, sim_agent, real_robot_interface):
        self.sim_agent = sim_agent
        self.real_robot = real_robot_interface
        self.domain_randomization_params = {}
        self.system_identification_params = {}

    def domain_randomization(self):
        """Apply domain randomization to improve sim-to-real transfer"""
        # Randomize simulation parameters
        self.domain_randomization_params = {
            'mass_variance': np.random.uniform(0.8, 1.2),      # ±20% mass variation
            'friction_variance': np.random.uniform(0.5, 1.5),  # 0.5x to 1.5x friction
            'actuator_noise': np.random.uniform(0.01, 0.05),   # 1-5% actuator noise
            'sensor_noise': np.random.uniform(0.001, 0.01),    # sensor noise levels
            'dynamics_randomization': np.random.uniform(0.9, 1.1)  # dynamics scaling
        }

    def system_identification(self) -> Dict[str, float]:
        """Identify system parameters for real robot"""
        # This would involve exciting the real robot and identifying parameters
        # For this example, we'll return placeholder values
        return {
            'mass': 50.0,  # kg
            'inertia': [1.0, 1.0, 1.0],  # kg*m^2
            'com_offset': [0.0, 0.0, 0.0],  # m
            'actuator_gains': [1.0, 1.0, 1.0]  # unitless
        }

    def adapt_policy(self, real_params: Dict[str, float]) -> torch.nn.Module:
        """Adapt simulation policy for real robot"""
        # Create adapted policy that accounts for real robot differences
        adapted_agent = self.sim_agent  # In practice, this would involve parameter adjustment

        # Update agent with real robot parameters
        # This might involve fine-tuning or parameter adjustment

        return adapted_agent

    def validate_transfer(self, adapted_agent) -> Dict[str, float]:
        """Validate policy transfer on real robot"""
        # This would run the adapted policy on the real robot
        # and measure performance
        return {
            'success_rate': 0.7,  # Example success rate
            'safety_violations': 0.05,  # 5% safety violation rate
            'performance_drop': 0.15  # 15% performance drop from sim
        }

def main():
    """Main function for RL validation and transfer"""
    print("Starting RL policy validation and transfer...")

    # Example validation process
    # In practice, this would connect to actual trained agents and environments
    print("Policy validation and transfer framework initialized")
    print("Note: Full implementation requires trained agents and robot interfaces")

if __name__ == "__main__":
    main()
```

## Summary

Reinforcement learning provides a powerful framework for training humanoid robots to perform complex tasks through trial and error. Key components include:

- **Environment Design**: Creating appropriate state, action, and reward spaces for the specific task
- **Algorithm Selection**: Choosing suitable RL algorithms (PPO, SAC, TD3) based on task requirements
- **Training Pipeline**: Implementing efficient training loops with proper data collection and updates
- **Validation**: Ensuring trained policies are safe and performant before deployment
- **Sim-to-Real Transfer**: Techniques to bridge the reality gap between simulation and real robots

## Exercises

### Conceptual
1. Compare and contrast different RL algorithms (PPO, SAC, TD3) for humanoid robot control. What are the trade-offs between sample efficiency, stability, and performance?

### Logical
1. Analyze the reward function design for a humanoid walking task. How would you balance competing objectives like forward velocity, energy efficiency, and balance stability?

### Implementation
1. Implement a complete PPO training pipeline for a simple humanoid reaching task in Isaac Sim, including proper reward shaping, curriculum learning, and policy validation.