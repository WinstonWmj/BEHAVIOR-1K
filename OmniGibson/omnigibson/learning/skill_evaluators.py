"""
Skill-specific success evaluators for subtask-level evaluation.

Each skill type (e.g. "pick up from", "place in") can have multiple evaluation
modes (e.g. "simple", "advanced") with different reward/success criteria.

Usage:
    evaluator = create_skill_evaluator("pick up from", mode="advanced")
    evaluator.reset(robot_evaluator, subtask_info)
    for step in rollout:
        step_metrics = evaluator.step(robot_evaluator)
    print(evaluator.is_success, evaluator.summary)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch as th

logger = logging.getLogger("skill_evaluator")

# Gripper finger indices within joint_qpos (28-dim) for R1Pro
_LEFT_GRIPPER_SLICE = slice(24, 26)
_RIGHT_GRIPPER_SLICE = slice(26, 28)
_GRIPPER_OPEN_VALUE = 0.05


class BaseSkillEvaluator(ABC):
    """Base class for all skill-specific evaluators."""

    @abstractmethod
    def reset(self, evaluator: Any, subtask_info: dict) -> None:
        """Initialize tracking state at the start of a subtask rollout."""

    @abstractmethod
    def step(self, evaluator: Any) -> dict:
        """Called every env step. Returns per-step metrics dict."""

    @property
    @abstractmethod
    def is_success(self) -> bool:
        """Whether the subtask has been successfully completed."""

    @property
    @abstractmethod
    def summary(self) -> dict:
        """Final summary metrics for logging and saving."""


# ---------------------------------------------------------------------------
# Pick Up From — Simple (displacement-based)
# ---------------------------------------------------------------------------

class PickUpFromSimple(BaseSkillEvaluator):
    """
    Success = manipulated object moved >= `distance_threshold` from its initial position.
    Purely geometric, ignores how the object moved.
    """

    def __init__(self, distance_threshold: float = 0.3):
        self.distance_threshold = distance_threshold
        self._target_obj = None
        self._init_pos: Optional[th.Tensor] = None
        self._max_displacement = 0.0
        self._success = False

    def reset(self, evaluator, subtask_info):
        manip_ids = subtask_info.get("manipulating_object_id", [])
        self._target_obj = None
        self._init_pos = None
        self._max_displacement = 0.0
        self._success = False

        if manip_ids:
            self._target_obj = evaluator.find_scene_object(manip_ids[0])
            if self._target_obj is not None:
                self._init_pos = self._target_obj.get_position_orientation()[0].clone()
                logger.info(
                    f"[Simple] Tracking '{manip_ids[0]}' at {self._init_pos.tolist()}, "
                    f"threshold={self.distance_threshold}m"
                )

    def step(self, evaluator) -> dict:
        if self._target_obj is None or self._init_pos is None:
            return {"displacement": 0.0, "success": False}

        current_pos = self._target_obj.get_position_orientation()[0]
        disp = th.norm(current_pos - self._init_pos).item()
        self._max_displacement = max(self._max_displacement, disp)
        if disp >= self.distance_threshold:
            self._success = True

        return {
            "displacement": round(disp, 4),
            "max_displacement": round(self._max_displacement, 4),
            "success": self._success,
        }

    @property
    def is_success(self) -> bool:
        return self._success

    @property
    def summary(self) -> dict:
        return {
            "mode": "simple",
            "max_displacement": round(self._max_displacement, 4),
            "success": self._success,
            "distance_threshold": self.distance_threshold,
        }


# ---------------------------------------------------------------------------
# Pick Up From — Advanced (grasp + lift + stability)
# ---------------------------------------------------------------------------

class PickUpFromAdvanced(BaseSkillEvaluator):
    """
    Composite success criteria for 'pick up from':

    1. **Grasp detected** — at least one gripper finger closed significantly
       (finger qpos < `grasp_finger_threshold`, default ~0.01).
    2. **Object lifted** — object z-position increased by >= `height_threshold`
       above its initial z (default 0.15 m).
    3. **Stable hold** — while grasped AND lifted, the object z-velocity stays
       above `fall_velocity_limit` (i.e. not free-falling) for at least
       `stable_steps_required` consecutive steps.

    Success = all three conditions met simultaneously for `stable_steps_required`
    consecutive steps.

    Also provides a continuous `reward` signal per step:
        reward = lift_reward + grasp_reward + stability_bonus
    """

    def __init__(
        self,
        height_threshold: float = 0.15,
        grasp_finger_threshold: float = 0.01,
        stable_steps_required: int = 10,
        fall_velocity_limit: float = -0.01,
    ):
        self.height_threshold = height_threshold
        self.grasp_finger_threshold = grasp_finger_threshold
        self.stable_steps_required = stable_steps_required
        self.fall_velocity_limit = fall_velocity_limit

        self._target_obj = None
        self._surface_obj = None
        self._init_pos: Optional[th.Tensor] = None
        self._prev_z: Optional[float] = None

        self._max_height_gain = 0.0
        self._max_displacement = 0.0
        self._grasp_ever_detected = False
        self._lifted_ever = False
        self._consecutive_stable = 0
        self._success = False
        self._step_count = 0

    def reset(self, evaluator, subtask_info):
        manip_ids = subtask_info.get("manipulating_object_id", [])
        all_obj_ids = subtask_info.get("object_id", [])

        self._target_obj = None
        self._surface_obj = None
        self._init_pos = None
        self._prev_z = None
        self._max_height_gain = 0.0
        self._max_displacement = 0.0
        self._grasp_ever_detected = False
        self._lifted_ever = False
        self._consecutive_stable = 0
        self._success = False
        self._step_count = 0

        if manip_ids:
            self._target_obj = evaluator.find_scene_object(manip_ids[0])
            if self._target_obj is not None:
                self._init_pos = self._target_obj.get_position_orientation()[0].clone()
                self._prev_z = self._init_pos[2].item()

        surface_ids = [oid for oid in all_obj_ids if oid not in set(manip_ids)]
        if surface_ids:
            self._surface_obj = evaluator.find_scene_object(surface_ids[0])

        logger.info(
            f"[Advanced] target='{manip_ids}', surface='{surface_ids}', "
            f"height_thresh={self.height_threshold}m, "
            f"stable_steps={self.stable_steps_required}"
        )

    def _detect_grasp(self, evaluator) -> bool:
        """Check if either gripper has a finger closed below threshold."""
        joint_qpos = evaluator.robot.get_joint_positions()
        left_fingers = joint_qpos[_LEFT_GRIPPER_SLICE]
        right_fingers = joint_qpos[_RIGHT_GRIPPER_SLICE]
        return (
            left_fingers.min().item() < self.grasp_finger_threshold
            or right_fingers.min().item() < self.grasp_finger_threshold
        )

    def step(self, evaluator) -> dict:
        if self._target_obj is None or self._init_pos is None:
            return {"reward": 0.0, "success": False}

        self._step_count += 1
        current_pos = self._target_obj.get_position_orientation()[0]
        current_z = current_pos[2].item()
        init_z = self._init_pos[2].item()

        # --- metrics ---
        height_gain = current_z - init_z
        self._max_height_gain = max(self._max_height_gain, height_gain)

        disp = th.norm(current_pos - self._init_pos).item()
        self._max_displacement = max(self._max_displacement, disp)

        # --- conditions ---
        grasping = self._detect_grasp(evaluator)
        if grasping:
            self._grasp_ever_detected = True

        lifted = height_gain >= self.height_threshold
        if lifted:
            self._lifted_ever = True

        z_vel = current_z - self._prev_z if self._prev_z is not None else 0.0
        self._prev_z = current_z
        not_falling = z_vel > self.fall_velocity_limit

        # --- compound success ---
        if grasping and lifted and not_falling:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0

        if self._consecutive_stable >= self.stable_steps_required:
            self._success = True

        # --- continuous reward signal ---
        lift_reward = np.clip(height_gain / self.height_threshold, 0.0, 1.0)
        grasp_reward = 0.3 if grasping else 0.0
        stability_bonus = 0.2 if (grasping and lifted and not_falling) else 0.0
        reward = lift_reward + grasp_reward + stability_bonus

        return {
            "reward": round(reward, 4),
            "height_gain": round(height_gain, 4),
            "displacement": round(disp, 4),
            "grasping": grasping,
            "lifted": lifted,
            "not_falling": not_falling,
            "consecutive_stable": self._consecutive_stable,
            "success": self._success,
        }

    @property
    def is_success(self) -> bool:
        return self._success

    @property
    def summary(self) -> dict:
        return {
            "mode": "advanced",
            "max_height_gain": round(self._max_height_gain, 4),
            "max_displacement": round(self._max_displacement, 4),
            "grasp_ever_detected": self._grasp_ever_detected,
            "lifted_ever": self._lifted_ever,
            "success": self._success,
            "height_threshold": self.height_threshold,
            "stable_steps_required": self.stable_steps_required,
            "total_steps": self._step_count,
        }


# ---------------------------------------------------------------------------
# Move To — Simple (robot-to-target 2D proximity)
# ---------------------------------------------------------------------------

class MoveToSimple(BaseSkillEvaluator):
    """
    Success = robot base moves within `distance_threshold` (XY-plane) of the
    target object.

    "move to" is a navigation skill: the robot must travel to the vicinity of
    a goal object.  We measure 2D (XY) distance because the robot's base
    height is largely irrelevant for ground locomotion.
    """

    def __init__(self, distance_threshold: float = 0.5):
        self.distance_threshold = distance_threshold
        self._target_obj = None
        self._init_robot_pos: Optional[th.Tensor] = None
        self._init_dist: Optional[float] = None
        self._min_dist = float("inf")
        self._success = False
        self._robot_ref = None

    def reset(self, evaluator, subtask_info):
        obj_ids = subtask_info.get("object_id", [])
        self._target_obj = None
        self._init_robot_pos = None
        self._init_dist = None
        self._min_dist = float("inf")
        self._success = False
        self._robot_ref = evaluator.robot

        if obj_ids:
            self._target_obj = evaluator.find_scene_object(obj_ids[0])

        if self._target_obj is not None and self._robot_ref is not None:
            robot_pos = self._robot_ref.get_position_orientation()[0]
            target_pos = self._target_obj.get_position_orientation()[0]
            self._init_robot_pos = robot_pos[:2].clone()
            self._init_dist = th.norm(robot_pos[:2] - target_pos[:2]).item()
            self._min_dist = self._init_dist
            logger.info(
                f"[MoveTo-Simple] Tracking target '{obj_ids[0]}', "
                f"init_dist={self._init_dist:.3f}m, threshold={self.distance_threshold}m"
            )

    def step(self, evaluator) -> dict:
        if self._target_obj is None or self._robot_ref is None:
            return {"dist_to_target": float("inf"), "success": False}

        robot_pos = self._robot_ref.get_position_orientation()[0][:2]
        target_pos = self._target_obj.get_position_orientation()[0][:2]
        dist = th.norm(robot_pos - target_pos).item()
        self._min_dist = min(self._min_dist, dist)

        if dist <= self.distance_threshold:
            self._success = True

        return {
            "dist_to_target": round(dist, 4),
            "min_dist_to_target": round(self._min_dist, 4),
            "success": self._success,
        }

    @property
    def is_success(self) -> bool:
        return self._success

    @property
    def summary(self) -> dict:
        return {
            "mode": "simple",
            "init_dist": round(self._init_dist, 4) if self._init_dist is not None else None,
            "min_dist_to_target": round(self._min_dist, 4),
            "success": self._success,
            "distance_threshold": self.distance_threshold,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILL_EVALUATOR_REGISTRY = {
    "pick up from": {
        "simple": PickUpFromSimple,
        "advanced": PickUpFromAdvanced,
    },
    "move to": {
        "simple": MoveToSimple,
    },
}


def create_skill_evaluator(
    skill_description: str,
    mode: str = "simple",
    **kwargs,
) -> Optional[BaseSkillEvaluator]:
    """
    Factory function to create a skill evaluator.

    Args:
        skill_description: the skill name from annotated json (e.g. "pick up from").
        mode: "simple" or "advanced".
        **kwargs: passed to the evaluator constructor (thresholds, etc.).

    Returns:
        A BaseSkillEvaluator instance, or None if the skill/mode is not registered.
    """
    skill_map = SKILL_EVALUATOR_REGISTRY.get(skill_description)
    if skill_map is None:
        logger.warning(f"No evaluator registered for skill '{skill_description}'")
        return None
    cls = skill_map.get(mode)
    if cls is None:
        logger.warning(f"No '{mode}' evaluator for skill '{skill_description}'")
        return None
    return cls(**kwargs)
