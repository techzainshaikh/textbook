---
title: LLM-Based Task Planning for Robotics
sidebar_position: 4
description: Implementing Large Language Model-based planning systems for humanoid robots to decompose high-level commands into executable action sequences
keywords: [LLM planning, task decomposition, robotics, humanoid AI, natural language processing, action sequences]
---

# Chapter 3: LLM-Based Task Planning for Robotics

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement LLM-based planning systems that translate high-level commands into executable robot actions
- Design task decomposition algorithms using Large Language Models for robotics applications
- Integrate LLM planning with robot execution frameworks and action servers
- Validate and verify LLM-generated plans for safety and correctness in robotic systems
- Handle planning failures and implement fallback strategies for robust robot operation

## Prerequisites

Students should have:
- Understanding of robotics concepts and action execution (covered in Module 1)
- Knowledge of Large Language Models and their capabilities
- Familiarity with task planning and decomposition concepts
- Basic understanding of natural language processing
- Experience with Python programming and API integration

## Core Concepts

Large Language Model (LLM)-based planning leverages the reasoning and decomposition capabilities of modern AI models to bridge high-level human commands with low-level robot actions. This approach enables more natural interaction with robots while maintaining the precision required for robotic execution.

### LLM Planning Architecture

**Command Interpretation:**
- **Natural Language Understanding**: Parse high-level commands and identify intent
- **Context Extraction**: Extract relevant entities, objects, locations, and constraints
- **Goal Specification**: Convert natural language into structured goal representations

**Task Decomposition:**
- **Hierarchical Planning**: Break down complex tasks into subtasks
- **Action Sequencing**: Order actions to achieve the desired goal
- **Constraint Handling**: Consider physical, temporal, and safety constraints

**Execution Integration:**
- **Action Mapping**: Translate LLM-generated steps to robot-specific actions
- **Feedback Integration**: Incorporate sensor feedback into the planning process
- **Plan Adaptation**: Adjust plans based on execution outcomes and environmental changes

### Planning Patterns

**Sequential Planning**: Linear execution of actions from start to goal
**Conditional Planning**: Decision points based on environmental conditions
**Reactive Planning**: Real-time adjustments based on sensor feedback
**Hierarchical Planning**: Multi-level decomposition from high-level goals to primitive actions

## Implementation

Let's implement LLM-based task planning for humanoid robotics:

### LLM Planning Interface

```python
#!/usr/bin/env python3
# llm_planning_interface.py

import openai
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class PlanStep:
    """Represents a single step in a robot plan"""
    action: str
    parameters: Dict[str, Any]
    description: str
    dependencies: List[str]  # Other steps this step depends on
    estimated_duration: float  # In seconds

@dataclass
class PlanningResult:
    """Result from the LLM planning process"""
    success: bool
    plan: List[PlanStep]
    reasoning: str
    confidence: float
    execution_context: Dict[str, Any]
    error: Optional[str] = None

class LLMPlanner:
    """
    Large Language Model-based planner for robotics
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo"):
        self.model = model
        if api_key:
            openai.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def create_planning_prompt(self, command: str, robot_capabilities: List[str],
                              environment_state: Dict[str, Any]) -> str:
        """Create a prompt for the LLM to generate a plan"""
        capabilities_str = ", ".join(robot_capabilities)
        env_state_str = json.dumps(environment_state, indent=2)

        prompt = f"""
You are an expert robot task planner. Your job is to decompose high-level commands into detailed, executable steps for a humanoid robot.

Robot Capabilities: {capabilities_str}

Current Environment State:
{env_state_str}

Command: {command}

Please generate a detailed plan with the following requirements:
1. Break down the command into specific, executable actions
2. Consider the robot's capabilities and environmental constraints
3. Include necessary preconditions and postconditions for each step
4. Order the steps logically for successful execution
5. Include error handling and fallback strategies where appropriate

Return your response in JSON format with the following structure:
{{
  "reasoning": "Your step-by-step reasoning for the plan",
  "plan": [
    {{
      "action": "action_name",
      "parameters": {{"param1": "value1", "param2": "value2"}},
      "description": "Brief description of what this step does",
      "dependencies": ["action_name_1", "action_name_2"],  // Actions that must complete before this one
      "estimated_duration": 2.5  // Estimated time in seconds
    }}
  ],
  "confidence": 0.9  // Confidence level in the plan (0.0 to 1.0)
}}

Ensure all actions are from the robot's capabilities list and all parameters are valid for those actions.
"""
        return prompt

    async def generate_plan(self, command: str, robot_capabilities: List[str],
                           environment_state: Dict[str, Any]) -> PlanningResult:
        """Generate a plan using the LLM"""
        try:
            prompt = self.create_planning_prompt(command, robot_capabilities, environment_state)

            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert robot task planner. Generate detailed, executable plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                # Convert to PlanStep objects
                plan_steps = []
                for step_data in plan_data.get("plan", []):
                    step = PlanStep(
                        action=step_data["action"],
                        parameters=step_data.get("parameters", {}),
                        description=step_data.get("description", ""),
                        dependencies=step_data.get("dependencies", []),
                        estimated_duration=step_data.get("estimated_duration", 1.0)
                    )
                    plan_steps.append(step)

                return PlanningResult(
                    success=True,
                    plan=plan_steps,
                    reasoning=plan_data.get("reasoning", ""),
                    confidence=plan_data.get("confidence", 0.5),
                    execution_context={
                        "original_command": command,
                        "robot_capabilities": robot_capabilities,
                        "environment_state": environment_state
                    }
                )
            else:
                return PlanningResult(
                    success=False,
                    plan=[],
                    reasoning="",
                    confidence=0.0,
                    execution_context={},
                    error=f"Could not extract JSON from LLM response: {response_text}"
                )

        except json.JSONDecodeError as e:
            return PlanningResult(
                success=False,
                plan=[],
                reasoning="",
                confidence=0.0,
                execution_context={},
                error=f"JSON decode error: {str(e)}"
            )
        except Exception as e:
            return PlanningResult(
                success=False,
                plan=[],
                reasoning="",
                confidence=0.0,
                execution_context={},
                error=f"Planning error: {str(e)}"
            )

    def validate_plan(self, plan: List[PlanStep], robot_capabilities: List[str]) -> Tuple[bool, List[str]]:
        """Validate that the plan is executable with available capabilities"""
        errors = []

        for step in plan:
            if step.action not in robot_capabilities:
                errors.append(f"Action '{step.action}' not available in robot capabilities")

            # Check parameter validity (simplified)
            if not isinstance(step.parameters, dict):
                errors.append(f"Parameters for action '{step.action}' must be a dictionary")

        return len(errors) == 0, errors

class RobotActionMapper:
    """
    Map LLM-generated actions to actual robot commands
    """

    def __init__(self):
        # Define mapping from high-level actions to robot-specific commands
        self.action_mapping = {
            "navigate_to": "move_base",
            "pick_object": "grasp",
            "place_object": "place",
            "grasp": "gripper_control",
            "release": "gripper_control",
            "inspect": "sensor_control",
            "approach": "move_base",
            "avoid": "move_base",
            "wait": "wait",
            "turn": "rotate"
        }

        # Define parameter mappings
        self.parameter_mappings = {
            "location": "target_pose",
            "object": "target_object",
            "position": "target_position",
            "orientation": "target_orientation",
            "gripper_position": "gripper_position"
        }

    def map_action(self, llm_action: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Map LLM action to robot action with appropriate parameters"""
        robot_action = self.action_mapping.get(llm_action, llm_action)
        robot_params = {}

        for param_name, param_value in parameters.items():
            robot_param_name = self.parameter_mappings.get(param_name, param_name)
            robot_params[robot_param_name] = param_value

        # Add default parameters if needed
        if robot_action == "gripper_control":
            if "gripper_position" not in robot_params:
                robot_params["gripper_position"] = 0.8 if llm_action == "grasp" else 0.0

        return robot_action, robot_params

class ExecutionValidator:
    """
    Validate and verify robot execution plans
    """

    def __init__(self):
        self.safety_constraints = [
            self._check_collision_avoidance,
            self._check_joint_limits,
            self._check_payload_limits,
            self._check_workspace_bounds
        ]

    def validate_for_execution(self, plan: List[PlanStep],
                              current_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate plan for safe execution"""
        errors = []

        for step in plan:
            for constraint_check in self.safety_constraints:
                is_safe, error_msg = constraint_check(step, current_state)
                if not is_safe:
                    errors.append(f"Step '{step.action}': {error_msg}")

        return len(errors) == 0, errors

    def _check_collision_avoidance(self, step: PlanStep, current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the action could cause collisions"""
        # This would interface with the robot's collision detection system
        # For demonstration, assume all actions are safe
        return True, ""

    def _check_joint_limits(self, step: PlanStep, current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the action respects joint limits"""
        # This would check robot joint limits
        return True, ""

    def _check_payload_limits(self, step: PlanStep, current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the action respects payload limits"""
        # This would check if the robot can handle the payload
        return True, ""

    def _check_workspace_bounds(self, step: PlanStep, current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the action is within workspace bounds"""
        # This would check if the target position is within workspace
        return True, ""

class AdaptivePlanner:
    """
    Adaptive planning that can adjust based on execution feedback
    """

    def __init__(self):
        self.action_mapper = RobotActionMapper()
        self.validator = ExecutionValidator()

    def adapt_plan(self, original_plan: List[PlanStep],
                   execution_feedback: Dict[str, Any]) -> List[PlanStep]:
        """Adapt the plan based on execution feedback"""
        # This would modify the plan based on what happened during execution
        # For example, if an object wasn't found, it might add a search step
        adapted_plan = []

        for step in original_plan:
            # Check if this step needs adaptation based on feedback
            if self._needs_adaptation(step, execution_feedback):
                adapted_steps = self._generate_adaptation(step, execution_feedback)
                adapted_plan.extend(adapted_steps)
            else:
                adapted_plan.append(step)

        return adapted_plan

    def _needs_adaptation(self, step: PlanStep, feedback: Dict[str, Any]) -> bool:
        """Determine if a step needs adaptation based on feedback"""
        # Check if the previous execution of similar steps failed
        failed_actions = feedback.get("failed_actions", [])
        return step.action in failed_actions

    def _generate_adaptation(self, step: PlanStep, feedback: Dict[str, Any]) -> List[PlanStep]:
        """Generate adapted steps for a failed action"""
        # This would generate alternative approaches
        # For example, if picking failed, try a different grasp strategy
        if step.action == "pick_object":
            # Add a search step before picking
            search_step = PlanStep(
                action="search_object",
                parameters=step.parameters,
                description="Search for the object before attempting to pick it",
                dependencies=step.dependencies,
                estimated_duration=2.0
            )
            return [search_step, step]

        return [step]

def create_llm_planner(api_key: str = None) -> LLMPlanner:
    """Factory function to create an LLM planner"""
    return LLMPlanner(api_key)
```

### Advanced Planning with Context and Memory

```python
#!/usr/bin/env python3
# advanced_planning.py

import openai
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class PlanExecutionRecord:
    """Record of a plan execution"""
    plan_id: str
    command: str
    plan: List[Dict[str, Any]]
    execution_time: float
    success: bool
    feedback: Dict[str, Any]
    timestamp: datetime

class PlanningMemory:
    """
    Memory system for learning from past planning experiences
    """

    def __init__(self):
        self.execution_history = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.max_history = 100

    def record_execution(self, record: PlanExecutionRecord):
        """Record a plan execution for future learning"""
        self.execution_history.append(record)

        # Keep only recent history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]

        # Update patterns based on success/failure
        if record.success:
            self._update_success_patterns(record)
        else:
            self._update_failure_patterns(record)

    def _update_success_patterns(self, record: PlanExecutionRecord):
        """Update patterns for successful executions"""
        command_key = self._normalize_command(record.command)
        if command_key not in self.success_patterns:
            self.success_patterns[command_key] = []

        self.success_patterns[command_key].append({
            "plan_structure": [step["action"] for step in record.plan],
            "environment": record.feedback.get("environment", {}),
            "execution_time": record.execution_time
        })

    def _update_failure_patterns(self, record: PlanExecutionRecord):
        """Update patterns for failed executions"""
        command_key = self._normalize_command(record.command)
        if command_key not in self.failure_patterns:
            self.failure_patterns[command_key] = []

        self.failure_patterns[command_key].append({
            "plan_structure": [step["action"] for step in record.plan],
            "failure_reason": record.feedback.get("error", "unknown"),
            "environment": record.feedback.get("environment", {})
        })

    def _normalize_command(self, command: str) -> str:
        """Normalize command for pattern matching"""
        return command.lower().strip()

    def get_successful_variants(self, command: str) -> List[Dict[str, Any]]:
        """Get successful plan variants for similar commands"""
        command_key = self._normalize_command(command)
        # This would use similarity matching in a real implementation
        return self.success_patterns.get(command_key, [])

    def get_failure_warnings(self, command: str) -> List[str]:
        """Get warnings based on past failures"""
        command_key = self._normalize_command(command)
        failures = self.failure_patterns.get(command_key, [])
        return [f["failure_reason"] for f in failures]

class ContextAwarePlanner:
    """
    Planner that uses context and memory to improve planning
    """

    def __init__(self, api_key: str = None):
        self.llm_planner = LLMPlanner(api_key)
        self.memory = PlanningMemory()
        self.action_mapper = RobotActionMapper()
        self.validator = ExecutionValidator()

    def create_contextual_prompt(self, command: str, robot_capabilities: List[str],
                                environment_state: Dict[str, Any]) -> str:
        """Create a prompt that includes contextual information"""
        # Get relevant past experiences
        successful_variants = self.memory.get_successful_variants(command)
        failure_warnings = self.memory.get_failure_warnings(command)

        capabilities_str = ", ".join(robot_capabilities)
        env_state_str = json.dumps(environment_state, indent=2)

        # Format successful variants
        variants_str = ""
        if successful_variants:
            variants_str = "Successful variants from similar commands:\n"
            for i, variant in enumerate(successful_variants[:2]):  # Limit to 2 examples
                variants_str += f"Variant {i+1}: {variant['plan_structure']}\n"

        # Format failure warnings
        warnings_str = ""
        if failure_warnings:
            warnings_str = "Potential failure points to avoid:\n"
            for warning in failure_warnings[:3]:  # Limit to 3 warnings
                warnings_str += f"- {warning}\n"

        prompt = f"""
You are an expert robot task planner. Your job is to decompose high-level commands into detailed, executable steps for a humanoid robot.

Robot Capabilities: {capabilities_str}

Current Environment State:
{env_state_str}

{variants_str}
{warnings_str}

Command: {command}

Please generate a detailed plan with the following requirements:
1. Break down the command into specific, executable actions
2. Consider the robot's capabilities and environmental constraints
3. Include necessary preconditions and postconditions for each step
4. Order the steps logically for successful execution
5. Include error handling and fallback strategies where appropriate
6. Consider lessons from past successful executions and failures

Return your response in JSON format with the following structure:
{{
  "reasoning": "Your step-by-step reasoning for the plan, considering past experiences",
  "plan": [
    {{
      "action": "action_name",
      "parameters": {{"param1": "value1", "param2": "value2"}},
      "description": "Brief description of what this step does",
      "dependencies": ["action_name_1", "action_name_2"],  // Actions that must complete before this one
      "estimated_duration": 2.5  // Estimated time in seconds
    }}
  ],
  "confidence": 0.9  // Confidence level in the plan (0.0 to 1.0)
}}

Ensure all actions are from the robot's capabilities list and all parameters are valid for those actions.
"""
        return prompt

    async def generate_contextual_plan(self, command: str, robot_capabilities: List[str],
                                     environment_state: Dict[str, Any]) -> PlanningResult:
        """Generate a plan using contextual information from memory"""
        try:
            prompt = self.create_contextual_prompt(command, robot_capabilities, environment_state)

            response = await openai.ChatCompletion.acreate(
                model=self.llm_planner.model,
                messages=[
                    {"role": "system", "content": "You are an expert robot task planner. Generate detailed, executable plans considering past experiences."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                # Convert to PlanStep objects
                plan_steps = []
                for step_data in plan_data.get("plan", []):
                    step = PlanStep(
                        action=step_data["action"],
                        parameters=step_data.get("parameters", {}),
                        description=step_data.get("description", ""),
                        dependencies=step_data.get("dependencies", []),
                        estimated_duration=step_data.get("estimated_duration", 1.0)
                    )
                    plan_steps.append(step)

                return PlanningResult(
                    success=True,
                    plan=plan_steps,
                    reasoning=plan_data.get("reasoning", ""),
                    confidence=plan_data.get("confidence", 0.5),
                    execution_context={
                        "original_command": command,
                        "robot_capabilities": robot_capabilities,
                        "environment_state": environment_state
                    }
                )
            else:
                return PlanningResult(
                    success=False,
                    plan=[],
                    reasoning="",
                    confidence=0.0,
                    execution_context={},
                    error=f"Could not extract JSON from LLM response: {response_text}"
                )

        except json.JSONDecodeError as e:
            return PlanningResult(
                success=False,
                plan=[],
                reasoning="",
                confidence=0.0,
                execution_context={},
                error=f"JSON decode error: {str(e)}"
            )
        except Exception as e:
            return PlanningResult(
                success=False,
                plan=[],
                reasoning="",
                confidence=0.0,
                execution_context={},
                error=f"Planning error: {str(e)}"
            )

    async def execute_plan_with_learning(self, command: str, robot_capabilities: List[str],
                                       environment_state: Dict[str, Any]) -> PlanExecutionRecord:
        """Execute a plan and record the experience for learning"""
        start_time = time.time()

        # Generate plan
        result = await self.generate_contextual_plan(command, robot_capabilities, environment_state)

        if not result.success:
            execution_time = time.time() - start_time
            record = PlanExecutionRecord(
                plan_id=f"plan_{int(time.time())}",
                command=command,
                plan=[],
                execution_time=execution_time,
                success=False,
                feedback={"error": result.error},
                timestamp=datetime.now()
            )
            self.memory.record_execution(record)
            return record

        # Validate plan
        is_valid, validation_errors = self.validator.validate_for_execution(
            result.plan, environment_state
        )

        if not is_valid:
            execution_time = time.time() - start_time
            record = PlanExecutionRecord(
                plan_id=f"plan_{int(time.time())}",
                command=command,
                plan=[step.__dict__ for step in result.plan],
                execution_time=execution_time,
                success=False,
                feedback={"validation_errors": validation_errors},
                timestamp=datetime.now()
            )
            self.memory.record_execution(record)
            return record

        # In a real system, this would execute the plan on the robot
        # For simulation, we'll assume successful execution
        execution_time = time.time() - start_time
        record = PlanExecutionRecord(
            plan_id=f"plan_{int(time.time())}",
            command=command,
            plan=[step.__dict__ for step in result.plan],
            execution_time=execution_time,
            success=True,
            feedback={"environment": environment_state},
            timestamp=datetime.now()
        )
        self.memory.record_execution(record)

        return record

def create_contextual_planner(api_key: str = None) -> ContextAwarePlanner:
    """Factory function to create a contextual planner"""
    return ContextAwarePlanner(api_key)
```

## Examples

### Example 1: Complex Task Planning System

```python
#!/usr/bin/env python3
# complex_task_planning.py

import asyncio
from typing import Dict, Any, List
import time

class ComplexTaskPlanningSystem:
    """
    System for planning complex tasks with multiple constraints and dependencies
    """

    def __init__(self, api_key: str = None):
        self.contextual_planner = create_contextual_planner(api_key)
        self.robot_capabilities = [
            "navigate_to", "pick_object", "place_object", "grasp", "release",
            "inspect", "approach", "wait", "turn", "detect_object"
        ]

    async def plan_complex_task(self, command: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a complex task with multiple constraints"""
        print(f"Planning complex task: {command}")

        # Generate initial plan
        result = await self.contextual_planner.generate_contextual_plan(
            command, self.robot_capabilities, environment_state
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "reasoning": result.reasoning
            }

        # Validate the plan
        is_valid, errors = self.contextual_planner.validator.validate_for_execution(
            result.plan, environment_state
        )

        if not is_valid:
            return {
                "success": False,
                "error": f"Plan validation failed: {errors}",
                "reasoning": result.reasoning
            }

        # Map high-level actions to robot-specific commands
        mapped_plan = []
        for step in result.plan:
            robot_action, robot_params = self.contextual_planner.action_mapper.map_action(
                step.action, step.parameters
            )
            mapped_plan.append({
                "robot_action": robot_action,
                "parameters": robot_params,
                "description": step.description,
                "estimated_duration": step.estimated_duration
            })

        return {
            "success": True,
            "plan": mapped_plan,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "estimated_total_time": sum(step.estimated_duration for step in result.plan)
        }

    async def execute_with_monitoring(self, command: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan with real-time monitoring and adaptation"""
        # Generate and execute plan
        execution_record = await self.contextual_planner.execute_plan_with_learning(
            command, self.robot_capabilities, environment_state
        )

        # Simulate execution monitoring
        monitoring_results = {
            "start_time": time.time(),
            "estimated_duration": execution_record.execution_time,
            "steps_completed": len(execution_record.plan) if execution_record.success else 0,
            "success": execution_record.success
        }

        return {
            "execution_record": execution_record,
            "monitoring": monitoring_results
        }

    async def handle_multi_object_task(self, objects_to_process: List[Dict[str, Any]],
                                     base_command_template: str) -> List[Dict[str, Any]]:
        """Handle tasks that involve multiple objects"""
        results = []

        for obj in objects_to_process:
            command = base_command_template.format(object_name=obj["name"], location=obj["location"])
            environment_state = {
                "available_objects": [obj],
                "robot_position": obj.get("robot_start_position", [0, 0, 0]),
                "workspace_limits": obj.get("workspace_limits", {})
            }

            result = await self.plan_complex_task(command, environment_state)
            results.append({
                "object": obj,
                "command": command,
                "plan_result": result
            })

        return results

# Example usage
async def main():
    """Main function to demonstrate complex task planning"""
    print("Initializing Complex Task Planning System...")

    # In a real implementation, you would provide an API key
    # For this example, we'll just demonstrate the structure
    try:
        planner_system = ComplexTaskPlanningSystem()

        # Example 1: Simple navigation task
        environment_state = {
            "objects": [{"name": "red_cube", "location": [1.0, 0.5, 0.0]}],
            "robot_position": [0, 0, 0],
            "workspace_limits": {"x": [-2, 2], "y": [-2, 2], "z": [0, 1]}
        }

        result = await planner_system.plan_complex_task(
            "Navigate to the red cube and pick it up",
            environment_state
        )

        print(f"Planning result: {result['success']}")
        if result['success']:
            print(f"Plan confidence: {result['confidence']:.2f}")
            print(f"Estimated time: {result['estimated_total_time']:.2f}s")

        print("Complex task planning system demonstrated successfully!")
        print("In a real implementation, this would connect to an LLM API and robot execution system.")

    except Exception as e:
        print(f"Note: This example requires an LLM API key for full functionality. Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Planning with Safety and Recovery

```python
#!/usr/bin/env python3
# planning_with_safety.py

import asyncio
import time
from typing import Dict, Any, List
import logging

class SafePlanningSystem:
    """
    Planning system with built-in safety checks and recovery mechanisms
    """

    def __init__(self, api_key: str = None):
        self.contextual_planner = create_contextual_planner(api_key)
        self.robot_capabilities = [
            "navigate_to", "pick_object", "place_object", "grasp", "release",
            "inspect", "approach", "wait", "turn", "detect_object", "emergency_stop"
        ]
        self.logger = logging.getLogger(__name__)

    def add_safety_constraints(self, plan: List[PlanStep]) -> List[PlanStep]:
        """Add safety checks to each step of the plan"""
        enhanced_plan = []

        for i, step in enumerate(plan):
            # Add safety check before critical actions
            if step.action in ["navigate_to", "pick_object", "place_object"]:
                # Add sensor check before action
                sensor_check = PlanStep(
                    action="inspect",
                    parameters={"target": step.parameters.get("location", step.parameters.get("object"))},
                    description=f"Verify environment is safe before {step.action}",
                    dependencies=step.dependencies[:],  # Copy dependencies
                    estimated_duration=0.5
                )
                enhanced_plan.append(sensor_check)

            # Add the original step
            enhanced_plan.append(step)

            # Add verification step after critical actions
            if step.action in ["grasp", "place_object", "navigate_to"]:
                verification = PlanStep(
                    action="inspect",
                    parameters={"target": step.parameters.get("object", step.parameters.get("location"))},
                    description=f"Verify successful completion of {step.action}",
                    dependencies=[step.action],  # Depends on the previous step
                    estimated_duration=0.5
                )
                enhanced_plan.append(verification)

        return enhanced_plan

    def create_recovery_plan(self, failed_step: PlanStep, error_type: str) -> List[PlanStep]:
        """Create a recovery plan for a failed step"""
        recovery_steps = []

        if error_type == "object_not_found":
            # Search for the object in nearby locations
            recovery_steps.extend([
                PlanStep(
                    action="inspect",
                    parameters={"search_area": "nearby"},
                    description="Search for object in nearby locations",
                    dependencies=[],
                    estimated_duration=2.0
                ),
                PlanStep(
                    action=failed_step.action,
                    parameters=failed_step.parameters,
                    description=f"Retry {failed_step.action} with updated object location",
                    dependencies=["inspect"],
                    estimated_duration=failed_step.estimated_duration
                )
            ])
        elif error_type == "collision_detected":
            # Plan alternative route
            recovery_steps.extend([
                PlanStep(
                    action="navigate_to",
                    parameters={"location": "safe_position", "avoid_obstacles": True},
                    description="Move to safe position to avoid collision",
                    dependencies=[],
                    estimated_duration=1.5
                ),
                PlanStep(
                    action=failed_step.action,
                    parameters=failed_step.parameters,
                    description=f"Retry {failed_step.action} with collision avoidance",
                    dependencies=["navigate_to"],
                    estimated_duration=failed_step.estimated_duration
                )
            ])
        elif error_type == "grasp_failed":
            # Try alternative grasp strategy
            recovery_steps.extend([
                PlanStep(
                    action="adjust_gripper",
                    parameters={"width": "wider"},
                    description="Adjust gripper for wider grasp",
                    dependencies=[],
                    estimated_duration=0.5
                ),
                PlanStep(
                    action=failed_step.action,
                    parameters=failed_step.parameters,
                    description=f"Retry {failed_step.action} with adjusted gripper",
                    dependencies=["adjust_gripper"],
                    estimated_duration=failed_step.estimated_duration
                )
            ])
        else:
            # General recovery - wait and retry
            recovery_steps.extend([
                PlanStep(
                    action="wait",
                    parameters={"duration": 1.0},
                    description="Wait before retrying failed action",
                    dependencies=[],
                    estimated_duration=1.0
                ),
                PlanStep(
                    action=failed_step.action,
                    parameters=failed_step.parameters,
                    description=f"Retry {failed_step.action}",
                    dependencies=["wait"],
                    estimated_duration=failed_step.estimated_duration
                )
            ])

        return recovery_steps

    async def plan_with_safety(self, command: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan with safety checks and recovery options"""
        # Generate initial plan
        result = await self.contextual_planner.generate_contextual_plan(
            command, self.robot_capabilities, environment_state
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "safety_enhanced_plan": [],
                "recovery_options": {}
            }

        # Add safety constraints
        safety_enhanced_plan = self.add_safety_constraints(result.plan)

        # Validate the enhanced plan
        is_valid, errors = self.contextual_planner.validator.validate_for_execution(
            safety_enhanced_plan, environment_state
        )

        if not is_valid:
            return {
                "success": False,
                "error": f"Enhanced plan validation failed: {errors}",
                "safety_enhanced_plan": [],
                "recovery_options": {}
            }

        # Generate recovery options for each step
        recovery_options = {}
        for step in safety_enhanced_plan:
            if step.action in ["navigate_to", "pick_object", "grasp", "place_object"]:
                # Common failure modes for these actions
                recovery_options[step.action] = {
                    "object_not_found": self.create_recovery_plan(step, "object_not_found"),
                    "collision_detected": self.create_recovery_plan(step, "collision_detected"),
                    "grasp_failed": self.create_recovery_plan(step, "grasp_failed"),
                    "general_failure": self.create_recovery_plan(step, "general")
                }

        return {
            "success": True,
            "original_plan": [s.__dict__ for s in result.plan],
            "safety_enhanced_plan": [s.__dict__ for s in safety_enhanced_plan],
            "recovery_options": recovery_options,
            "estimated_duration": sum(s.estimated_duration for s in safety_enhanced_plan),
            "confidence": result.confidence
        }

    async def simulate_execution_with_recovery(self, command: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution with potential failures and recovery"""
        # Plan with safety
        plan_result = await self.plan_with_safety(command, environment_state)

        if not plan_result["success"]:
            return plan_result

        # Simulate execution (in a real system, this would interface with the robot)
        execution_log = []
        current_step = 0
        total_time = 0

        for step in plan_result["safety_enhanced_plan"]:
            execution_log.append({
                "step": step,
                "status": "executing",
                "timestamp": time.time()
            })

            # Simulate step execution time
            await asyncio.sleep(step.get("estimated_duration", 0.1))
            total_time += step.get("estimated_duration", 0.1)

            # Simulate potential failure for critical actions
            if step["action"] in ["navigate_to", "pick_object"] and current_step % 3 == 2:
                # Simulate a failure every 3rd critical action
                execution_log[-1]["status"] = "failed"
                execution_log[-1]["error"] = "object_not_found"

                # Apply recovery
                recovery_plan = plan_result["recovery_options"].get(step["action"], {}).get("object_not_found", [])
                if recovery_plan:
                    execution_log.append({
                        "step": "recovery",
                        "recovery_plan": [r.__dict__ for r in recovery_plan],
                        "status": "applying_recovery",
                        "timestamp": time.time()
                    })

                    # Simulate recovery execution
                    for recovery_step in recovery_plan:
                        await asyncio.sleep(recovery_step.estimated_duration)
                        total_time += recovery_step.estimated_duration

                    execution_log.append({
                        "step": "recovery",
                        "status": "recovery_successful",
                        "timestamp": time.time()
                    })

            else:
                execution_log[-1]["status"] = "completed"

            current_step += 1

        return {
            "execution_log": execution_log,
            "total_time": total_time,
            "final_status": "completed_with_recovery" if any(log.get("status") == "applying_recovery" for log in execution_log) else "completed_successfully"
        }

def main():
    """Main function for safe planning demonstration"""
    print("Initializing Safe Planning System...")

    # In a real implementation, you would provide an API key
    # For this example, we'll just demonstrate the structure
    print("Safe planning system initialized with safety checks and recovery mechanisms.")
    print("The system includes:")
    print("- Safety constraints added to critical actions")
    print("- Recovery plans for common failure modes")
    print("- Real-time monitoring and adaptation capabilities")
    print("- Comprehensive error handling and fallback strategies")

if __name__ == "__main__":
    main()
```

## Summary

LLM-based task planning represents a significant advancement in robotics, enabling natural language interaction with complex robotic systems. Key aspects include:

- **Natural Language Interface**: Users can express complex tasks in plain English
- **Intelligent Decomposition**: LLMs break down high-level goals into executable steps
- **Context Awareness**: Planning considers environment state and past experiences
- **Safety Integration**: Built-in safety checks and recovery mechanisms
- **Adaptive Learning**: Systems improve over time through experience

## Exercises

### Conceptual
1. Compare the advantages and limitations of LLM-based planning versus classical planning algorithms (e.g., PDDL, STRIPS) for robotics applications. When would you choose one approach over the other?

### Logical
1. Design a planning system that can handle ambiguous commands (e.g., "clean the room" when the specific objects and cleaning methods are not specified). How would your system determine the appropriate level of detail and specific actions?

### Implementation
1. Implement an LLM-based planning system that integrates with ROS 2 action servers, including proper error handling, safety validation, and plan adaptation based on execution feedback.