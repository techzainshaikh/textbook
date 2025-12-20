---
title: Sim-to-Real Transfer for Humanoid Robots
sidebar_position: 7
description: Bridging the reality gap between simulation and real-world humanoid robot deployment using domain randomization and system identification
keywords: [sim-to-real, domain randomization, system identification, reality gap, humanoid robotics, transfer learning]
---

# Chapter 6: Sim-to-Real Transfer for Humanoid Robots

## Learning Objectives

By the end of this chapter, students will be able to:
- Identify and analyze the reality gap between simulation and real-world robotics
- Implement domain randomization techniques to improve policy robustness
- Perform system identification for real robot parameter estimation
- Apply transfer learning methods to adapt simulation-trained policies for real robots
- Validate and deploy policies from simulation to real humanoid robots safely

## Prerequisites

Students should have:
- Understanding of simulation environments (covered in Module 2)
- Knowledge of reinforcement learning concepts (covered in Chapter 5)
- Experience with robot control and dynamics (covered in Module 1)
- Basic understanding of probability and statistics for domain randomization
- Familiarity with system identification techniques

## Core Concepts

Sim-to-real transfer addresses the challenge of deploying policies trained in simulation on real robots. The reality gap arises from differences in dynamics, sensors, actuators, and environmental conditions between simulation and reality.

### The Reality Gap

**Sources of Discrepancy:**
- **Dynamics Modeling**: Inaccuracies in mass, inertia, friction, and contact models
- **Sensor Noise**: Differences in sensor characteristics, noise patterns, and delays
- **Actuator Dynamics**: Motor response times, gear backlash, and control delays
- **Environmental Conditions**: Surface properties, lighting, and external disturbances
- **Hardware Limitations**: Joint limits, power constraints, and mechanical wear

**Transfer Approaches:**
- **Domain Randomization**: Randomizing simulation parameters to improve robustness
- **System Identification**: Measuring real robot parameters for simulation calibration
- **Adaptive Control**: Adjusting control policies based on real-world performance
- **Domain Adaptation**: Learning mappings between simulation and real data
- **Few-Shot Learning**: Rapid adaptation with minimal real-world data

### Domain Randomization Techniques

Domain randomization involves randomizing simulation parameters across wide ranges to force the policy to learn robust features that work across different conditions.

**Physical Parameters:**
- Mass scaling (±20%)
- Friction coefficients (0.1x to 10x)
- Actuator delays and noise
- Sensor noise and bias
- Contact stiffness and damping

**Visual Parameters:**
- Lighting conditions and directions
- Texture variations
- Color and contrast changes
- Camera noise and distortion
- Environmental appearance

## Implementation

Let's implement sim-to-real transfer techniques for humanoid robot deployment:

### Domain Randomization Environment

```python
#!/usr/bin/env python3
# domain_randomization.py

import numpy as np
import torch
import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class DomainRandomizationParams:
    """Parameters for domain randomization"""
    mass_range: Tuple[float, float] = (0.8, 1.2)  # ±20% mass variation
    friction_range: Tuple[float, float] = (0.1, 2.0)  # 0.1x to 2x friction
    actuator_noise_range: Tuple[float, float] = (0.001, 0.01)  # Actuator noise
    sensor_noise_range: Tuple[float, float] = (0.001, 0.01)  # Sensor noise
    dynamics_randomization: Tuple[float, float] = (0.9, 1.1)  # Dynamics scaling
    delay_range: Tuple[float, float] = (0.0, 0.05)  # Actuator delays in seconds

class DomainRandomizedEnvironment:
    """
    Environment with domain randomization for sim-to-real transfer
    """

    def __init__(self, base_env, randomization_params: DomainRandomizationParams):
        self.base_env = base_env
        self.params = randomization_params
        self.current_randomization = self.randomize_parameters()
        self.episode_count = 0

    def randomize_parameters(self) -> Dict[str, float]:
        """Randomize simulation parameters"""
        randomization = {
            'mass_multiplier': random.uniform(*self.params.mass_range),
            'friction_multiplier': random.uniform(*self.params.friction_range),
            'actuator_noise_std': random.uniform(*self.params.actuator_noise_range),
            'sensor_noise_std': random.uniform(*self.params.sensor_noise_range),
            'dynamics_multiplier': random.uniform(*self.params.dynamics_randomization),
            'actuator_delay': random.uniform(*self.params.delay_range)
        }
        return randomization

    def reset(self) -> np.ndarray:
        """Reset environment with new randomization parameters"""
        if self.episode_count > 0 and self.episode_count % 10 == 0:  # Randomize every 10 episodes
            self.current_randomization = self.randomize_parameters()
            self.update_simulation_parameters()

        self.episode_count += 1
        return self.base_env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step with randomized parameters"""
        # Add actuator noise
        noisy_action = self.add_actuator_noise(action)

        # Apply actuator delay simulation
        delayed_action = self.apply_actuator_delay(noisy_action)

        # Execute step in base environment
        obs, reward, done, info = self.base_env.step(delayed_action)

        # Add sensor noise to observation
        noisy_obs = self.add_sensor_noise(obs)

        # Update info with randomization parameters
        info['randomization_params'] = self.current_randomization

        return noisy_obs, reward, done, info

    def add_actuator_noise(self, action: np.ndarray) -> np.ndarray:
        """Add noise to actuator commands"""
        noise = np.random.normal(0, self.current_randomization['actuator_noise_std'], size=action.shape)
        return action + noise

    def apply_actuator_delay(self, action: np.ndarray) -> np.ndarray:
        """Simulate actuator delay"""
        # In a real implementation, this would maintain a delay buffer
        return action  # Simplified for this example

    def add_sensor_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add noise to sensor observations"""
        noise = np.random.normal(0, self.current_randomization['sensor_noise_std'], size=observation.shape)
        return observation + noise

    def update_simulation_parameters(self):
        """Update simulation with current randomization parameters"""
        # This would interface with the physics simulator to update parameters
        # For example, updating mass, friction, etc. of robot links
        pass

class CurriculumDomainRandomization:
    """
    Progressive domain randomization that increases difficulty over time
    """

    def __init__(self, base_env, initial_params: DomainRandomizationParams,
                 max_params: DomainRandomizationParams, curriculum_steps: int = 1000):
        self.base_env = base_env
        self.initial_params = initial_params
        self.max_params = max_params
        self.curriculum_steps = curriculum_steps
        self.current_step = 0

    def get_current_params(self) -> DomainRandomizationParams:
        """Get current domain randomization parameters based on curriculum progress"""
        progress = min(1.0, self.current_step / self.curriculum_steps)

        # Interpolate between initial and max parameters
        current_params = DomainRandomizationParams()

        # Interpolate mass range
        initial_mass_range = self.initial_params.mass_range
        max_mass_range = self.max_params.mass_range
        current_params.mass_range = (
            initial_mass_range[0] + progress * (max_mass_range[0] - initial_mass_range[0]),
            initial_mass_range[1] + progress * (max_mass_range[1] - initial_mass_range[1])
        )

        # Interpolate friction range
        initial_friction_range = self.initial_params.friction_range
        max_friction_range = self.max_params.friction_range
        current_params.friction_range = (
            initial_friction_range[0] + progress * (max_friction_range[0] - initial_friction_range[0]),
            initial_friction_range[1] + progress * (max_friction_range[1] - initial_friction_range[1])
        )

        # Continue for other parameters...
        current_params.actuator_noise_range = (
            self.initial_params.actuator_noise_range[0] +
            progress * (self.max_params.actuator_noise_range[0] - self.initial_params.actuator_noise_range[0]),
            self.initial_params.actuator_noise_range[1] +
            progress * (self.max_params.actuator_noise_range[1] - self.initial_params.actuator_noise_range[1])
        )

        current_params.sensor_noise_range = (
            self.initial_params.sensor_noise_range[0] +
            progress * (self.max_params.sensor_noise_range[0] - self.initial_params.sensor_noise_range[0]),
            self.initial_params.sensor_noise_range[1] +
            progress * (self.max_params.sensor_noise_range[1] - self.initial_params.sensor_noise_range[1])
        )

        return current_params

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step with curriculum-based randomization"""
        # Update curriculum progress
        self.current_step += 1

        # Get current parameters
        current_params = self.get_current_params()

        # Create domain randomized environment with current parameters
        dr_env = DomainRandomizedEnvironment(self.base_env, current_params)

        # Execute step
        return dr_env.step(action)

def create_domain_randomized_env(base_env):
    """Factory function to create domain randomized environment"""
    initial_params = DomainRandomizationParams(
        mass_range=(0.9, 1.1),  # Start with ±10%
        friction_range=(0.5, 2.0),  # Start with 0.5x to 2x
        actuator_noise_range=(0.001, 0.005),  # Start with lower noise
        sensor_noise_range=(0.001, 0.005)   # Start with lower noise
    )

    max_params = DomainRandomizationParams(
        mass_range=(0.5, 2.0),  # End with ±50% to 2x
        friction_range=(0.1, 5.0),  # End with 0.1x to 5x
        actuator_noise_range=(0.005, 0.05),  # End with higher noise
        sensor_noise_range=(0.005, 0.05)   # End with higher noise
    )

    return CurriculumDomainRandomization(base_env, initial_params, max_params)
```

### System Identification for Real Robot Calibration

```python
#!/usr/bin/env python3
# system_identification.py

import numpy as np
from scipy.optimize import minimize
from scipy import signal
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import control  # python-control package

class SystemIdentifier:
    """
    System identification for real robot parameter estimation
    """

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.identified_params = {}
        self.excitation_signals = []
        self.measured_responses = []

    def generate_excitation_signal(self, duration: float, frequency_range: Tuple[float, float],
                                   amplitude: float, sampling_rate: int = 100) -> np.ndarray:
        """Generate excitation signal for system identification"""
        t = np.linspace(0, duration, int(duration * sampling_rate))

        # Generate multi-sine signal covering frequency range
        signal_components = []
        freqs = np.logspace(np.log10(frequency_range[0]), np.log10(frequency_range[1]), 10)

        for freq in freqs:
            component = amplitude * np.sin(2 * np.pi * freq * t)
            signal_components.append(component)

        # Add random phases
        excitation_signal = np.sum(signal_components, axis=0)

        # Add some random noise for robustness
        noise = np.random.normal(0, amplitude * 0.1, size=excitation_signal.shape)
        excitation_signal += noise

        return excitation_signal

    def collect_data(self, joint_indices: List[int], duration: float = 10.0) -> Dict[str, np.ndarray]:
        """Collect input-output data for system identification"""
        data = {
            'time': [],
            'inputs': {idx: [] for idx in joint_indices},
            'outputs': {idx: [] for idx in joint_indices},
            'velocities': {idx: [] for idx in joint_indices},
            'accelerations': {idx: [] for idx in joint_indices}
        }

        # Generate excitation for each joint
        for joint_idx in joint_indices:
            # Generate excitation signal
            excitation = self.generate_excitation_signal(duration, (0.1, 10.0), 0.1)

            # Apply excitation and collect response
            for i, u in enumerate(excitation):
                # Apply input to joint
                self.robot_model.apply_torque(joint_idx, u)

                # Step simulation
                self.robot_model.step()

                # Collect data
                data['time'].append(i / len(excitation) * duration)
                data['inputs'][joint_idx].append(u)
                data['outputs'][joint_idx].append(self.robot_model.get_joint_position(joint_idx))
                data['velocities'][joint_idx].append(self.robot_model.get_joint_velocity(joint_idx))

                # Calculate acceleration (approximate)
                if i > 1:
                    dt = duration / len(excitation)
                    prev_vel = self.robot_model.get_joint_velocity(joint_idx, prev=True)
                    acc = (self.robot_model.get_joint_velocity(joint_idx) - prev_vel) / dt
                    data['accelerations'][joint_idx].append(acc)
                else:
                    data['accelerations'][joint_idx].append(0.0)

        return data

    def identify_mass_matrix(self, data: Dict[str, np.ndarray], joint_indices: List[int]) -> np.ndarray:
        """Identify mass matrix using collected data"""
        # For each joint, estimate the mass based on input-output relationship
        # This is a simplified approach - real system identification would be more complex

        n_joints = len(joint_indices)
        mass_matrix = np.zeros((n_joints, n_joints))

        for i, joint_idx in enumerate(joint_indices):
            # Extract relevant data
            inputs = np.array(data['inputs'][joint_idx])
            accelerations = np.array(data['accelerations'][joint_idx])

            # Estimate mass: F = M*a => M = F/a (for single DOF)
            # In reality, this would involve more complex MIMO system identification
            valid_mask = np.abs(accelerations) > 1e-6  # Avoid division by small numbers

            if np.any(valid_mask):
                estimated_mass = np.mean(inputs[valid_mask] / accelerations[valid_mask])
                mass_matrix[i, i] = estimated_mass
            else:
                mass_matrix[i, i] = 1.0  # Default value

        return mass_matrix

    def identify_friction_parameters(self, data: Dict[str, np.ndarray], joint_indices: List[int]) -> Dict[int, Dict[str, float]]:
        """Identify friction parameters (Coulomb and viscous)"""
        friction_params = {}

        for joint_idx in joint_indices:
            velocities = np.array(data['velocities'][joint_idx])
            torques = np.array(data['inputs'][joint_idx])

            # Model: tau_friction = tau_coulomb * sign(v) + tau_viscous * v
            # Use least squares to fit friction model
            A = np.column_stack([np.sign(velocities), velocities])

            try:
                # Solve: min ||A*x - torques||^2
                params, residuals, rank, s = np.linalg.lstsq(A, torques, rcond=None)
                coulomb_friction, viscous_friction = params

                friction_params[joint_idx] = {
                    'coulomb': abs(coulomb_friction),
                    'viscous': abs(viscous_friction)
                }
            except:
                # Default values if identification fails
                friction_params[joint_idx] = {
                    'coulomb': 0.1,
                    'viscous': 0.01
                }

        return friction_params

    def identify_compliance(self, data: Dict[str, np.ndarray], joint_indices: List[int]) -> Dict[int, float]:
        """Identify joint compliance/stiffness parameters"""
        compliance_params = {}

        for joint_idx in joint_indices:
            positions = np.array(data['outputs'][joint_idx])
            torques = np.array(data['inputs'][joint_idx])

            # Estimate stiffness: k = delta_tau / delta_theta
            # Remove mean to focus on variations
            pos_var = positions - np.mean(positions)
            tau_var = torques - np.mean(torques)

            # Use least squares to estimate stiffness
            if np.var(pos_var) > 1e-10:  # Avoid division by small variance
                stiffness = np.cov(tau_var, pos_var)[0, 1] / np.var(pos_var)
                compliance_params[joint_idx] = 1.0 / abs(stiffness) if abs(stiffness) > 1e-6 else 1e6
            else:
                compliance_params[joint_idx] = 1e-6  # Very stiff (low compliance)

        return compliance_params

    def identify_full_model(self, joint_indices: List[int]) -> Dict[str, Any]:
        """Perform complete system identification"""
        print("Collecting data for system identification...")
        data = self.collect_data(joint_indices)

        print("Identifying mass matrix...")
        mass_matrix = self.identify_mass_matrix(data, joint_indices)

        print("Identifying friction parameters...")
        friction_params = self.identify_friction_parameters(data, joint_indices)

        print("Identifying compliance parameters...")
        compliance_params = self.identify_compliance(data, joint_indices)

        # Compile results
        results = {
            'mass_matrix': mass_matrix,
            'friction_params': friction_params,
            'compliance_params': compliance_params,
            'data_used': data
        }

        self.identified_params = results
        return results

    def update_simulation_model(self, sim_env, identified_params: Dict[str, Any]):
        """Update simulation environment with identified parameters"""
        # Update mass properties
        for i, joint_idx in enumerate(identified_params['mass_matrix']):
            sim_env.update_joint_mass(joint_idx, identified_params['mass_matrix'][i, i])

        # Update friction parameters
        for joint_idx, friction_info in identified_params['friction_params'].items():
            sim_env.update_joint_friction(joint_idx,
                                        friction_info['coulomb'],
                                        friction_info['viscous'])

        # Update compliance parameters
        for joint_idx, compliance in identified_params['compliance_params'].items():
            sim_env.update_joint_compliance(joint_idx, compliance)

def validate_identification(identified_model, real_robot, test_trajectories: List[np.ndarray]):
    """Validate system identification results"""
    errors = []

    for trajectory in test_trajectories:
        # Apply same inputs to both models
        real_response = apply_trajectory(real_robot, trajectory)
        identified_response = apply_trajectory(identified_model, trajectory)

        # Calculate error
        error = np.mean(np.abs(real_response - identified_response))
        errors.append(error)

    return np.mean(errors)

def apply_trajectory(robot, trajectory: np.ndarray) -> np.ndarray:
    """Apply trajectory to robot and return response"""
    # This is a placeholder function
    # In practice, this would apply the trajectory and collect the response
    pass
```

### Policy Adaptation and Transfer

```python
#!/usr/bin/env python3
# policy_adaptation.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import copy

class PolicyAdapter:
    """
    Adapt simulation-trained policies for real robot deployment
    """

    def __init__(self, sim_policy: nn.Module, real_robot_params: Dict[str, Any]):
        self.sim_policy = sim_policy
        self.real_params = real_robot_params
        self.adapted_policy = None
        self.adaptation_network = None

    def create_adaptation_network(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Create network to adapt simulation policy to real robot"""
        # Adaptation network learns the residual between sim and real
        self.adaptation_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),  # State and sim action as input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Output action adjustment
            nn.Tanh()  # Keep adjustments bounded
        )

    def adapt_policy(self, real_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    learning_rate: float = 1e-4, epochs: int = 100):
        """Adapt policy using real robot data"""
        if self.adaptation_network is None:
            raise ValueError("Adaptation network not created. Call create_adaptation_network first.")

        optimizer = torch.optim.Adam(self.adaptation_network.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0

            for state, sim_action, real_action in real_data:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                sim_action_tensor = torch.FloatTensor(sim_action).unsqueeze(0)
                real_action_tensor = torch.FloatTensor(real_action).unsqueeze(0)

                # Get sim policy action
                with torch.no_grad():
                    sim_action_pred = self.sim_policy(state_tensor)[0]  # Assuming policy returns (action, value)

                # Get adaptation adjustment
                adaptation_input = torch.cat([state_tensor, sim_action_pred], dim=1)
                action_adjustment = self.adaptation_network(adaptation_input)

                # Combined action
                adapted_action = sim_action_pred + action_adjustment

                # Loss: difference between adapted action and real action
                loss = nn.MSELoss()(adapted_action, real_action_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Adaptation epoch {epoch}, Loss: {total_loss/len(real_data):.6f}")

        # Create adapted policy
        self.adapted_policy = self.create_adapted_policy()

    def create_adapted_policy(self):
        """Create a policy that combines sim policy with adaptation network"""
        class AdaptedPolicy(nn.Module):
            def __init__(self, sim_policy, adaptation_network):
                super().__init__()
                self.sim_policy = sim_policy
                self.adaptation_network = adaptation_network

            def forward(self, state):
                # Get action from sim policy
                sim_action, value = self.sim_policy(state)

                # Get adaptation adjustment
                adaptation_input = torch.cat([state, sim_action], dim=1)
                action_adjustment = self.adaptation_network(adaptation_input)

                # Return adapted action
                adapted_action = sim_action + action_adjustment
                return adapted_action, value

        return AdaptedPolicy(self.sim_policy, self.adaptation_network)

    def get_adapted_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from adapted policy"""
        if self.adapted_policy is None:
            # If not adapted, return sim policy action
            with torch.no_grad():
                action, _ = self.sim_policy(torch.FloatTensor(state).unsqueeze(0))
            return action.squeeze(0).numpy()

        with torch.no_grad():
            action, _ = self.adapted_policy(torch.FloatTensor(state).unsqueeze(0))
        return action.squeeze(0).numpy()

class FineTuningAdapter:
    """
    Fine-tune simulation policy using real robot data
    """

    def __init__(self, sim_policy: nn.Module, learning_rate: float = 1e-5):
        self.sim_policy = copy.deepcopy(sim_policy)
        self.optimizer = torch.optim.Adam(self.sim_policy.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def fine_tune(self, real_data: List[Tuple[np.ndarray, np.ndarray]],
                 epochs: int = 50, batch_size: int = 32):
        """Fine-tune policy using real robot data"""
        for epoch in range(epochs):
            # Shuffle data
            shuffled_data = real_data.copy()
            np.random.shuffle(shuffled_data)

            total_loss = 0
            num_batches = len(shuffled_data) // batch_size

            for i in range(num_batches):
                batch = shuffled_data[i*batch_size:(i+1)*batch_size]

                states = torch.FloatTensor([item[0] for item in batch])
                actions = torch.FloatTensor([item[1] for item in batch])

                # Get policy predictions
                pred_actions, _ = self.sim_policy(states)

                # Calculate loss
                loss = self.criterion(pred_actions, actions)

                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Fine-tuning epoch {epoch}, Average Loss: {avg_loss:.6f}")

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from fine-tuned policy"""
        with torch.no_grad():
            action, _ = self.sim_policy(torch.FloatTensor(state).unsqueeze(0))
        return action.squeeze(0).numpy()

class DomainAdversarialAdapter:
    """
    Use domain adversarial training to adapt policies
    """

    def __init__(self, policy: nn.Module, state_dim: int):
        self.policy = policy
        self.domain_classifier = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-4)
        self.domain_criterion = nn.BCELoss()

    def train_adversarial(self, sim_data: List[Tuple[np.ndarray, np.ndarray]],
                         real_data: List[Tuple[np.ndarray, np.ndarray]],
                         epochs: int = 100):
        """Train using domain adversarial approach"""
        for epoch in range(epochs):
            # Train domain classifier
            self.domain_optimizer.zero_grad()

            # Sim data (label 0)
            sim_states = torch.FloatTensor([item[0] for item in sim_data])
            sim_labels = torch.zeros(sim_states.size(0), 1)

            # Real data (label 1)
            real_states = torch.FloatTensor([item[0] for item in real_data])
            real_labels = torch.ones(real_states.size(0), 1)

            # Combine and shuffle
            all_states = torch.cat([sim_states, real_states], dim=0)
            all_labels = torch.cat([sim_labels, real_labels], dim=0)

            domain_preds = self.domain_classifier(all_states)
            domain_loss = self.domain_criterion(domain_preds, all_labels)

            domain_loss.backward()
            self.domain_optimizer.step()

            # Train policy to fool domain classifier (gradient reversal)
            self.policy_optimizer.zero_grad()

            # Get features from policy (before action output)
            # This assumes the policy has a feature extractor
            sim_features = self.get_policy_features(sim_states)
            real_features = self.get_policy_features(real_states)

            # Domain classifier should not be able to distinguish
            sim_domain_pred = self.domain_classifier(sim_features)
            real_domain_pred = self.domain_classifier(real_features)

            # We want domain classifier to output 0.5 for both (unconfident)
            sim_adv_loss = self.domain_criterion(sim_domain_pred, torch.ones_like(sim_domain_pred) * 0.5)
            real_adv_loss = self.domain_criterion(real_domain_pred, torch.ones_like(real_domain_pred) * 0.5)

            adv_loss = sim_adv_loss + real_adv_loss
            adv_loss.backward()
            self.policy_optimizer.step()

            if epoch % 20 == 0:
                print(f"Adversarial training epoch {epoch}, Domain Loss: {domain_loss.item():.6f}, Adv Loss: {adv_loss.item():.6f}")

    def get_policy_features(self, states):
        """Extract features from policy (implementation depends on policy architecture)"""
        # This is a placeholder - actual implementation depends on policy structure
        # For a typical policy, this might be the output of the feature extractor
        # before the action and value heads
        return states  # Simplified for this example

def safety_wrapper(policy, safety_checker, state_filter=None):
    """
    Wrap policy with safety checks
    """
    def safe_policy(state):
        # Apply state filtering if provided
        if state_filter:
            state = state_filter(state)

        # Get action from policy
        action = policy(state)

        # Check safety
        if not safety_checker.is_safe(state, action):
            # Return safe fallback action
            return safety_checker.get_safe_action(state)

        return action

    return safe_policy

class SafetyChecker:
    """
    Safety checker for real robot deployment
    """

    def __init__(self, joint_limits: Dict[int, Tuple[float, float]],
                 max_velocities: Dict[int, Tuple[float, float]]):
        self.joint_limits = joint_limits
        self.max_velocities = max_velocities

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if action is safe given current state"""
        # Check joint limits (simplified)
        for joint_idx, (lower, upper) in self.joint_limits.items():
            if joint_idx < len(state):
                if state[joint_idx] < lower or state[joint_idx] > upper:
                    return False

        # Check action magnitude
        if np.any(np.abs(action) > 10.0):  # Arbitrary threshold
            return False

        return True

    def get_safe_action(self, state: np.ndarray) -> np.ndarray:
        """Get safe fallback action"""
        # Return zero action as safe fallback
        return np.zeros_like(state[:len(self.joint_limits)])
```

## Examples

### Example 1: Complete Sim-to-Real Pipeline

```python
#!/usr/bin/env python3
# complete_sim_to_real_pipeline.py

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class CompleteSimToRealPipeline:
    """
    Complete pipeline for sim-to-real transfer
    """

    def __init__(self, sim_env, real_robot):
        self.sim_env = sim_env
        self.real_robot = real_robot
        self.trained_policy = None
        self.adapted_policy = None

    def train_in_simulation(self, policy_architecture, training_params):
        """Train policy in simulation"""
        print("Training policy in simulation...")

        # Create and train policy in simulation environment
        # This would use the RL training methods from Chapter 5
        policy = policy_architecture

        # Training loop would go here
        # For this example, we'll assume training is completed
        self.trained_policy = policy

        print("Simulation training completed")

    def perform_system_identification(self, joint_indices: List[int]):
        """Perform system identification on real robot"""
        print("Performing system identification...")

        identifier = SystemIdentifier(self.real_robot)
        identified_params = identifier.identify_full_model(joint_indices)

        print("Updating simulation with identified parameters...")
        identifier.update_simulation_model(self.sim_env, identified_params)

        return identified_params

    def collect_real_data(self, num_episodes: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Collect data from real robot for adaptation"""
        real_data = []

        for episode in range(num_episodes):
            state = self.real_robot.reset()
            done = False

            while not done:
                # Get action from simulation policy (without noise for data collection)
                with torch.no_grad():
                    sim_action_tensor, _ = self.trained_policy(torch.FloatTensor(state).unsqueeze(0))
                    sim_action = sim_action_tensor.squeeze(0).numpy()

                # Apply action to real robot
                real_action = self.real_robot.apply_action(sim_action)
                next_state, reward, done, info = self.real_robot.step(real_action)

                # Store (state, sim_action, real_action) tuple
                real_data.append((state, sim_action, real_action))

                state = next_state

        return real_data

    def adapt_policy(self, real_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """Adapt simulation policy using real data"""
        print("Adapting policy with real data...")

        adapter = PolicyAdapter(self.trained_policy, {})
        adapter.create_adaptation_network(
            state_dim=self.sim_env.observation_space.shape[0],
            action_dim=self.sim_env.action_space.shape[0]
        )

        adapter.adapt_policy(real_data)
        self.adapted_policy = adapter

        print("Policy adaptation completed")

    def validate_adapted_policy(self, num_episodes: int = 5) -> Dict[str, float]:
        """Validate adapted policy on real robot"""
        print("Validating adapted policy...")

        success_count = 0
        total_reward = 0
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.real_robot.reset()
            episode_reward = 0
            step_count = 0
            done = False

            while not done and step_count < 1000:  # Max 1000 steps per episode
                # Get action from adapted policy
                action = self.adapted_policy.get_adapted_action(state)

                # Apply action to real robot
                next_state, reward, done, info = self.real_robot.step(action)

                state = next_state
                episode_reward += reward
                step_count += 1

            total_reward += episode_reward
            episode_lengths.append(step_count)

            # Check if task was successful (this would depend on specific task)
            if info.get('success', False):
                success_count += 1

        results = {
            'success_rate': success_count / num_episodes,
            'average_reward': total_reward / num_episodes,
            'average_length': np.mean(episode_lengths)
        }

        print(f"Validation results: {results}")
        return results

    def run_complete_pipeline(self, joint_indices: List[int], num_real_episodes: int = 20):
        """Run the complete sim-to-real pipeline"""
        print("Starting complete sim-to-real pipeline...")

        # Step 1: Train in simulation (assumed to be done)
        # self.train_in_simulation(policy_architecture, training_params)

        # Step 2: System identification
        identified_params = self.perform_system_identification(joint_indices)

        # Step 3: Collect real data
        real_data = self.collect_real_data(num_real_episodes)

        # Step 4: Adapt policy
        self.adapt_policy(real_data)

        # Step 5: Validate adapted policy
        validation_results = self.validate_adapted_policy()

        print("Complete sim-to-real pipeline finished!")
        return validation_results

def main():
    """Main function for complete sim-to-real pipeline"""
    # This would integrate with actual simulation and real robot interfaces
    print("Complete sim-to-real transfer pipeline framework initialized")
    print("Note: Full implementation requires actual sim and real robot interfaces")

if __name__ == "__main__":
    main()
```

### Example 2: Safety-Critical Deployment

```python
#!/usr/bin/env python3
# safety_critical_deployment.py

import numpy as np
from typing import Dict, Any, List, Tuple
import time
import threading
from contextlib import contextmanager

class SafetyManager:
    """
    Safety management for real robot deployment
    """

    def __init__(self, real_robot, max_torque_limits: Dict[int, float],
                 emergency_stop_callback=None):
        self.real_robot = real_robot
        self.max_torque_limits = max_torque_limits
        self.emergency_stop_callback = emergency_stop_callback
        self.is_safe_mode = True
        self.emergency_stop_triggered = False

        # Safety monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False

    def start_safety_monitoring(self):
        """Start safety monitoring in background thread"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._safety_monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_safety_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _safety_monitor_loop(self):
        """Continuous safety monitoring loop"""
        while self.monitoring_active:
            try:
                # Check joint limits
                if self._check_joint_limits():
                    self.trigger_emergency_stop("Joint limit violation")
                    continue

                # Check torque limits
                if self._check_torque_limits():
                    self.trigger_emergency_stop("Torque limit violation")
                    continue

                # Check velocity limits
                if self._check_velocity_limits():
                    self.trigger_emergency_stop("Velocity limit violation")
                    continue

                # Check for hardware faults
                if self._check_hardware_faults():
                    self.trigger_emergency_stop("Hardware fault detected")
                    continue

            except Exception as e:
                print(f"Safety monitoring error: {e}")
                self.trigger_emergency_stop("Safety system error")

            time.sleep(0.01)  # 100 Hz monitoring

    def _check_joint_limits(self) -> bool:
        """Check if any joints are out of safe limits"""
        # This would interface with real robot to check joint positions
        return False  # Placeholder

    def _check_torque_limits(self) -> bool:
        """Check if any joints exceed torque limits"""
        # This would interface with real robot to check joint torques
        return False  # Placeholder

    def _check_velocity_limits(self) -> bool:
        """Check if any joints exceed velocity limits"""
        # This would interface with real robot to check joint velocities
        return False  # Placeholder

    def _check_hardware_faults(self) -> bool:
        """Check for hardware faults"""
        # This would interface with real robot to check for faults
        return False  # Placeholder

    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        print(f"EMERGENCY STOP: {reason}")
        self.emergency_stop_triggered = True
        self.is_safe_mode = True

        # Stop all robot motion
        self.real_robot.emergency_stop()

        if self.emergency_stop_callback:
            self.emergency_stop_callback(reason)

    def is_system_safe(self) -> bool:
        """Check if system is in safe state"""
        return not self.emergency_stop_triggered and self.is_safe_mode

    @contextmanager
    def safe_execution(self):
        """Context manager for safe policy execution"""
        if not self.is_system_safe():
            raise RuntimeError("System not in safe state")

        try:
            self.is_safe_mode = False  # Allow execution
            yield
        finally:
            self.is_safe_mode = True   # Return to safe mode
            if self.emergency_stop_triggered:
                raise RuntimeError("Emergency stop was triggered during execution")

class DeploymentManager:
    """
    Manage safe deployment of adapted policies
    """

    def __init__(self, real_robot, safety_manager: SafetyManager):
        self.real_robot = real_robot
        self.safety_manager = safety_manager
        self.adapted_policy = None
        self.is_deployed = False

    def deploy_policy(self, adapted_policy, validation_threshold: float = 0.7):
        """Deploy policy after safety validation"""
        print("Starting policy deployment...")

        # Validate policy safety
        if not self._validate_policy_safety(adapted_policy, validation_threshold):
            raise ValueError("Policy did not pass safety validation")

        # Set the adapted policy
        self.adapted_policy = adapted_policy

        # Start safety monitoring
        self.safety_manager.start_safety_monitoring()

        self.is_deployed = True
        print("Policy deployed successfully")

    def _validate_policy_safety(self, policy, threshold: float) -> bool:
        """Validate policy safety before deployment"""
        print("Validating policy safety...")

        # Run safety tests with the policy
        safety_tests_passed = 0
        total_tests = 5

        for test_idx in range(total_tests):
            try:
                # Run a short safety test
                test_result = self._run_safety_test(policy)
                if test_result['success']:
                    safety_tests_passed += 1
            except:
                continue  # Test failed

        success_rate = safety_tests_passed / total_tests
        print(f"Safety validation: {success_rate:.2f} ({safety_tests_passed}/{total_tests})")

        return success_rate >= threshold

    def _run_safety_test(self, policy) -> Dict[str, Any]:
        """Run a safety test with the policy"""
        # Reset robot to safe position
        self.real_robot.reset_to_safe_position()

        # Run policy for a short duration
        state = self.real_robot.get_state()
        for step in range(50):  # 50 steps safety test
            action = policy.get_adapted_action(state)

            # Check if action is safe
            if not self._is_action_safe(state, action):
                return {'success': False, 'reason': 'Unsafe action detected'}

            # Apply action
            next_state, reward, done, info = self.real_robot.step(action)

            # Check for safety violations
            if self.safety_manager.emergency_stop_triggered:
                return {'success': False, 'reason': 'Emergency stop triggered'}

            state = next_state

            if done:
                break

        return {'success': True}

    def _is_action_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if action is safe for current state"""
        # Check torque limits
        if np.any(np.abs(action) > list(self.safety_manager.max_torque_limits.values())[0:action.shape[0]]):
            return False

        # Check for potential joint limit violations
        # This would involve forward simulation
        return True

    def execute_policy(self, max_duration: float = 60.0):
        """Execute deployed policy for specified duration"""
        if not self.is_deployed:
            raise ValueError("No policy deployed")

        if not self.safety_manager.is_system_safe():
            raise ValueError("System not in safe state")

        start_time = time.time()
        step_count = 0

        try:
            with self.safety_manager.safe_execution():
                state = self.real_robot.reset()

                while (time.time() - start_time) < max_duration:
                    if self.safety_manager.emergency_stop_triggered:
                        break

                    # Get action from adapted policy
                    action = self.adapted_policy.get_adapted_action(state)

                    # Apply action to robot
                    next_state, reward, done, info = self.real_robot.step(action)

                    state = next_state
                    step_count += 1

                    if done:
                        state = self.real_robot.reset()

                    # Small delay to allow safety systems to respond
                    time.sleep(0.001)

        except Exception as e:
            print(f"Policy execution error: {e}")
            self.safety_manager.trigger_emergency_stop(f"Execution error: {e}")

        execution_time = time.time() - start_time
        print(f"Policy executed for {execution_time:.2f}s ({step_count} steps)")

    def undeploy_policy(self):
        """Safely undeploy the policy"""
        self.safety_manager.stop_safety_monitoring()
        self.is_deployed = False
        self.adapted_policy = None
        print("Policy undeployed safely")

def main():
    """Main function for safety-critical deployment"""
    print("Safety-critical deployment system initialized")
    print("Note: Full implementation requires real robot interface and safety systems")

if __name__ == "__main__":
    main()
```

## Summary

Sim-to-real transfer is a critical step in deploying simulation-trained policies on real humanoid robots. Key components include:

- **Domain Randomization**: Randomizing simulation parameters to improve policy robustness across different conditions
- **System Identification**: Measuring real robot parameters to calibrate simulation models
- **Policy Adaptation**: Techniques to adapt simulation-trained policies for real robot characteristics
- **Safety Management**: Ensuring safe deployment and operation of learned policies on real robots
- **Validation**: Comprehensive testing to ensure policies perform safely and effectively on real hardware

## Exercises

### Conceptual
1. Explain the concept of the "reality gap" in robotics and describe three different approaches to address it.

### Logical
1. Analyze the trade-offs between using domain randomization versus system identification for sim-to-real transfer. When would you choose one approach over the other?

### Implementation
1. Implement a complete sim-to-real pipeline for a simple humanoid reaching task, including domain randomization, system identification, policy adaptation, and safe deployment on a simulated real robot.