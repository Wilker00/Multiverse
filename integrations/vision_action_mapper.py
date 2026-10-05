"""
integrations/vision_action_mapper.py

Vision-Action Mapper: Grounds game trajectories (pixels + discrete actions)
into robot embodiment spaces (joint angles + continuous control).

This enables transfer learning from game-trained agents (Atari, Grid, etc.)
to physical and embodied robots without embodiment-specific retraining.

Architecture:
1. SpatialConceptExtractor: Extract task-space motion intent from observations
2. ActionMorphism: Map discrete/game actions to continuous/robot commands
3. EmbodimentAdapter: Ground task-space velocities into joint control
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


@dataclasses.dataclass
class EmbodimentSpec:
    """Specification of a robot embodiment."""
    
    name: str  # "UR10", "Humanoid", "Drone", etc.
    dof: int  # Degrees of freedom
    action_space_low: np.ndarray  # Action lower bounds
    action_space_high: np.ndarray  # Action upper bounds
    action_type: str  # "velocity", "torque", "position"
    
    # Kinematic chain (optional, for IK)
    joint_names: Optional[List[str]] = None
    joint_limits_low: Optional[np.ndarray] = None
    joint_limits_high: Optional[np.ndarray] = None
    
    # End-effector frame (for task-space control)
    end_effector_name: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate spec consistency."""
        assert len(self.action_space_low) == self.dof
        assert len(self.action_space_high) == self.dof
        assert np.all(self.action_space_low <= self.action_space_high)
        if self.joint_limits_low is not None:
            assert len(self.joint_limits_low) == self.dof
            assert len(self.joint_limits_high) == self.dof
        return True


class SpatialConceptExtractor(ABC):
    """
    Extracts task-space motion intent from consecutive observations.
    
    Transforms observation sequences into normalized spatial concepts:
    - Position delta (Δx, Δy, Δz)
    - Velocity estimate
    - Direction to goal
    - Distance to goal
    """
    
    @abstractmethod
    def extract(self, obs_sequence: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract spatial concepts from observation sequence.
        
        Args:
            obs_sequence: List of observations [obs_t-1, obs_t, ..., obs_t+n]
        
        Returns:
            Dict with keys:
            - "delta_pos": (3,) array, normalized position change
            - "velocity": (3,) array, normalized velocity
            - "goal_vector": (3,) array, direction to goal
            - "goal_distance": float, normalized distance to goal
        """
        pass


class DefaultSpatialExtractor(SpatialConceptExtractor):
    """
    Default spatial extractor for flat observation spaces.
    
    Assumes observations are flat arrays where:
    - Last 2-3 dims are agent position (x, y) or (x, y, z)
    - Remaining dims (except last) may include target position or delta
    """
    
    def __init__(self, agent_pos_dims: Tuple[int, int] = (-2, -1)):
        """
        Args:
            agent_pos_dims: Indices of agent position in flat observation.
                Default (-2, -1) for (x, y) at end.
        """
        self.agent_pos_dims = agent_pos_dims
    
    def extract(self, obs_sequence: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract spatial concepts from observation sequence."""
        if len(obs_sequence) < 2:
            # Not enough data; return zero intent
            return {
                "delta_pos": np.array([0.0, 0.0, 0.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "goal_vector": np.array([0.0, 0.0, 1.0]),
                "goal_distance": 1.0,
            }
        
        obs_t = obs_sequence[-2]  # Previous
        obs_t_plus_1 = obs_sequence[-1]  # Current
        
        # Extract agent positions
        pos_t = self._extract_position(obs_t)
        pos_t1 = self._extract_position(obs_t_plus_1)
        
        # Compute delta and velocity
        delta_pos = pos_t1 - pos_t
        # Normalize to [-1, 1] assuming max reasonable speed is 1.0 per step
        delta_pos_norm = np.clip(delta_pos, -1.0, 1.0)
        
        velocity = delta_pos_norm  # Approximated from consecutive steps
        
        # Goal vector: assume it's in latter part of observation
        goal_vector = self._extract_goal_vector(obs_t_plus_1)
        goal_distance = np.linalg.norm(goal_vector) if len(goal_vector) > 0 else 1.0
        
        return {
            "delta_pos": delta_pos_norm,
            "velocity": velocity,
            "goal_vector": goal_vector / (goal_distance + 1e-6),
            "goal_distance": min(goal_distance, 1.0),  # Normalize
        }
    
    def _extract_position(self, obs: np.ndarray) -> np.ndarray:
        """Extract (x, y, z) position from flat observation."""
        pos_x = obs[self.agent_pos_dims[0]]
        pos_y = obs[self.agent_pos_dims[1]]
        pos_z = 0.0  # Assume 2D by default
        return np.array([pos_x, pos_y, pos_z])
    
    def _extract_goal_vector(self, obs: np.ndarray) -> np.ndarray:
        """Extract goal/target vector from observation."""
        # Heuristic: assume last 2-3 elements after agent position are goal
        if len(obs) >= 4:
            # (agent_x, agent_y, goal_x, goal_y)
            return np.array([obs[-2] - obs[self.agent_pos_dims[0]],
                           obs[-1] - obs[self.agent_pos_dims[1]],
                           0.0])
        return np.array([0.0, 0.0, 1.0])


class ActionMorphism:
    """
    Transforms discrete/game actions into continuous/robot actions.
    
    Models a morphism (structure-preserving map):
        f: A_discrete × S_game → A_continuous × S_robot
    
    Where discrete actions map to abstract intents (left, right, forward),
    and those intents are translated to continuous robot commands.
    """
    
    # Canonical intent types
    INTENT_FORWARD = "forward"
    INTENT_BACKWARD = "backward"
    INTENT_LEFT = "left"
    INTENT_RIGHT = "right"
    INTENT_UP = "up"
    INTENT_DOWN = "down"
    INTENT_STOP = "stop"
    
    def __init__(self, game_action_space_size: int = 4):
        """
        Args:
            game_action_space_size: Number of discrete actions in game.
                Default 4 (up, down, left, right for Atari-style).
        """
        self.game_action_space_size = game_action_space_size
        self._build_intent_map()
    
    def _build_intent_map(self) -> None:
        """Build mapping from discrete actions to abstract intents."""
        # Standard Atari / grid action mapping
        if self.game_action_space_size == 4:
            self.action_to_intent = {
                0: self.INTENT_UP,
                1: self.INTENT_DOWN,
                2: self.INTENT_LEFT,
                3: self.INTENT_RIGHT,
            }
        elif self.game_action_space_size == 3:
            self.action_to_intent = {
                0: self.INTENT_LEFT,
                1: self.INTENT_STOP,
                2: self.INTENT_RIGHT,
            }
        else:
            # Generic fallback
            self.action_to_intent = {i: f"action_{i}" for i in range(self.game_action_space_size)}
    
    def game_action_to_intent(self, action: int) -> str:
        """Convert discrete game action to abstract intent."""
        return self.action_to_intent.get(action, "stop")
    
    def intent_to_robot_velocity(self, intent: str, speed_scale: float = 1.0) -> np.ndarray:
        """
        Convert abstract intent to 6D task-space velocity (vx, vy, vz, ωx, ωy, ωz).
        
        Args:
            intent: Abstract intent (forward, left, right, etc.)
            speed_scale: Speed magnitude (0-1).
        
        Returns:
            (6,) velocity vector in task space.
        """
        base_speed = speed_scale * 0.5  # Base speed magnitude
        
        velocity = np.zeros(6)
        
        if intent == self.INTENT_FORWARD:
            velocity[0] = base_speed  # +X direction
        elif intent == self.INTENT_BACKWARD:
            velocity[0] = -base_speed
        elif intent == self.INTENT_LEFT:
            velocity[1] = base_speed  # +Y direction (left in standard frame)
        elif intent == self.INTENT_RIGHT:
            velocity[1] = -base_speed
        elif intent == self.INTENT_UP:
            velocity[2] = base_speed  # +Z direction
        elif intent == self.INTENT_DOWN:
            velocity[2] = -base_speed
        elif intent == self.INTENT_STOP:
            velocity[:] = 0.0
        
        return velocity
    
    def game_action_to_robot_velocity(self, action: int, speed_scale: float = 1.0) -> np.ndarray:
        """Convert game action directly to robot velocity."""
        intent = self.game_action_to_intent(action)
        return self.intent_to_robot_velocity(intent, speed_scale)


class EmbodimentAdapter:
    """
    Grounds task-space velocities (vx, vy, vz, ωx, ωy, ωz) into
    joint-space control commands for a specific embodiment.
    
    For now, implements a simple kinematic adapter:
    - For arms: basic inverse kinematics or joint velocity mapping
    - For mobile platforms: differential drive or omni-wheel mapping
    - For quadrotors: attitude + thrust decomposition
    """
    
    def __init__(self, embodiment_spec: EmbodimentSpec):
        self.spec = embodiment_spec
        self.spec.validate()
    
    def task_space_velocity_to_joint_commands(
        self,
        task_velocity: np.ndarray,  # (6,) vx, vy, vz, ωx, ωy, ωz
        current_joint_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert task-space velocity to joint-space control commands.
        
        Args:
            task_velocity: (6,) task-space velocity vector
            current_joint_state: (dof,) current joint angles (for IK)
        
        Returns:
            (dof,) joint control command in action space bounds
        """
        
        # Simple mapping: distribute task-space velocity to joints
        # For a 3-DOF arm: map (vx, vy, vz) to joint velocities
        if self.spec.dof == 2:
            # 2-link planar arm
            return self._2link_arm_adapter(task_velocity)
        elif self.spec.dof == 3:
            # 3-link arm or 3D point mass
            return self._3dof_adapter(task_velocity)
        elif self.spec.dof == 6:
            # 6-DOF arm (UR10-like)
            return self._6dof_arm_adapter(task_velocity)
        else:
            # Fallback: linear scaling to action bounds
            return self._default_adapter(task_velocity)
    
    def _2link_arm_adapter(self, task_velocity: np.ndarray) -> np.ndarray:
        """Simple adapter for 2-link planar arm."""
        # Map (vx, vy) to (θ1_dot, θ2_dot)
        # Simplified: each velocity component → joint velocity
        cmd = np.array([task_velocity[0], task_velocity[1]])
        return self._clip_to_bounds(cmd)
    
    def _3dof_adapter(self, task_velocity: np.ndarray) -> np.ndarray:
        """Adapter for 3-DOF system."""
        # Map (vx, vy, vz) to joint velocities
        cmd = task_velocity[:3]
        return self._clip_to_bounds(cmd)
    
    def _6dof_arm_adapter(self, task_velocity: np.ndarray) -> np.ndarray:
        """Adapter for 6-DOF arm."""
        # Distribute 6D task velocity across 6 joints
        # This is a simplified linear mapping; real IK would be more complex
        cmd = task_velocity[:6]
        return self._clip_to_bounds(cmd)
    
    def _default_adapter(self, task_velocity: np.ndarray) -> np.ndarray:
        """Fallback adapter for arbitrary DOF."""
        # Scale to match DOF
        if len(task_velocity) >= self.spec.dof:
            cmd = task_velocity[:self.spec.dof]
        else:
            cmd = np.concatenate([
                task_velocity,
                np.zeros(self.spec.dof - len(task_velocity))
            ])
        return self._clip_to_bounds(cmd)
    
    def _clip_to_bounds(self, cmd: np.ndarray) -> np.ndarray:
        """Clip command to embodiment action space bounds."""
        return np.clip(cmd, self.spec.action_space_low, self.spec.action_space_high)


class VisionActionMapper:
    """
    Complete vision-action mapper: orchestrates extraction → morphism → adaptation.
    
    Transforms game trajectories (pixels, discrete actions) into
    robot trajectories (flat obs, continuous commands).
    """
    
    def __init__(
        self,
        embodiment_spec: EmbodimentSpec,
        spatial_extractor: Optional[SpatialConceptExtractor] = None,
        action_morphism: Optional[ActionMorphism] = None,
    ):
        self.embodiment_spec = embodiment_spec
        self.spatial_extractor = spatial_extractor or DefaultSpatialExtractor()
        self.action_morphism = action_morphism or ActionMorphism(game_action_space_size=4)
        self.adapter = EmbodimentAdapter(embodiment_spec)
    
    def map_trajectory(
        self,
        game_obs_sequence: List[np.ndarray],
        game_actions: List[int],
        speed_scale: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Map a game trajectory to robot trajectory.
        
        Args:
            game_obs_sequence: List of game observations
            game_actions: List of discrete game actions
            speed_scale: Speed magnitude for continuous control (0-1)
        
        Returns:
            (robot_obs_sequence, robot_action_sequence)
            Both are sequences aligned in time.
        """
        robot_obs_sequence = []
        robot_action_sequence = []
        
        # Convert game observations to robot observation space
        for obs in game_obs_sequence:
            robot_obs = self._ground_observation(obs)
            robot_obs_sequence.append(robot_obs)
        
        # Convert game actions to robot actions
        for t, action in enumerate(game_actions):
            # Get context: current and next observation for motion intent
            obs_window = game_obs_sequence[max(0, t - 1):t + 2]
            
            # Extract spatial intent
            spatial_concepts = self.spatial_extractor.extract(obs_window)
            
            # Morphism: discrete action → task-space velocity
            task_velocity = self.action_morphism.game_action_to_robot_velocity(
                action, speed_scale=speed_scale
            )
            
            # Modulate by spatial concepts (e.g., slow down near goal)
            modulated_velocity = self._modulate_velocity(
                task_velocity, spatial_concepts
            )
            
            # Adapt to embodiment
            robot_action = self.adapter.task_space_velocity_to_joint_commands(
                modulated_velocity
            )
            
            robot_action_sequence.append(robot_action)
        
        return robot_obs_sequence, robot_action_sequence
    
    def _ground_observation(self, game_obs: np.ndarray) -> np.ndarray:
        """Convert game observation to flat robot observation."""
        # Simple case: assume game obs is already flat or can be flattened
        if isinstance(game_obs, (list, tuple)):
            obs_flat = np.array(game_obs, dtype=np.float32)
        else:
            obs_flat = np.asarray(game_obs, dtype=np.float32).flatten()
        
        # Normalize to [0, 1] or [-1, 1] range
        obs_flat = np.clip(obs_flat, -10.0, 10.0) / 10.0
        
        # Pad/truncate to match robot observation size
        # (in real system, this would map game space → robot sensor space)
        return obs_flat
    
    def _modulate_velocity(
        self,
        task_velocity: np.ndarray,
        spatial_concepts: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Modulate task-space velocity based on spatial concepts.
        
        E.g., reduce speed near goal to prevent overshoot.
        """
        modulated = task_velocity.copy()
        
        # Reduce speed as goal approaches
        goal_distance = spatial_concepts["goal_distance"]
        proximity_scale = 1.0 - (0.5 * (1.0 - goal_distance))
        modulated *= proximity_scale
        
        return modulated


# ============================================================================
# Convenience factories for common embodiments
# ============================================================================

def embodiment_ur10() -> EmbodimentSpec:
    """UR10 6-DOF collaborative robot."""
    return EmbodimentSpec(
        name="UR10",
        dof=6,
        action_space_low=np.array([-1.0] * 6),
        action_space_high=np.array([1.0] * 6),
        action_type="velocity",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        end_effector_name="tool0",
    )


def embodiment_3link_planar_arm() -> EmbodimentSpec:
    """3-link planar arm for testing."""
    return EmbodimentSpec(
        name="PlanarArm3Link",
        dof=3,
        action_space_low=np.array([-1.0] * 3),
        action_space_high=np.array([1.0] * 3),
        action_type="velocity",
        joint_names=["joint_1", "joint_2", "joint_3"],
    )


def embodiment_2link_planar_arm() -> EmbodimentSpec:
    """2-link planar arm (simplest for POC)."""
    return EmbodimentSpec(
        name="PlanarArm2Link",
        dof=2,
        action_space_low=np.array([-1.0, -1.0]),
        action_space_high=np.array([1.0, 1.0]),
        action_type="velocity",
        joint_names=["joint_1", "joint_2"],
    )


def embodiment_quadrotor() -> EmbodimentSpec:
    """Quadrotor drone."""
    return EmbodimentSpec(
        name="Quadrotor",
        dof=4,
        action_space_low=np.array([0.0] * 4),  # Motor thrust non-negative
        action_space_high=np.array([1.0] * 4),
        action_type="thrust",
        joint_names=["motor_1", "motor_2", "motor_3", "motor_4"],
    )
