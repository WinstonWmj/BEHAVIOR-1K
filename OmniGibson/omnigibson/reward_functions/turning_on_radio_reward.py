import json
import os
import re

import torch as th

from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.touching import Touching
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.toggle import ToggledOn, m as toggle_macros
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward


class TurningOnRadioReward(SequentialTaskReward):
    """
    Task-bound sequential reward for `turning_on_radio`.

    Reading guide:
    1. `reset()` resolves the target radio and the support object (prefer annotation, then fallback).
    2. `_build_stages()` defines the four sequential subtasks.
    3. `_evaluate_stage()` implements dense reward + completion logic for the active stage only.
    4. Helper methods below encode the robustness tweaks we needed during debugging:
       object identity matching, support-state fallbacks, and rich debug metrics.
    """

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=5.0,
        move_to_dense_scale=0.3,
        inhand_infer_distance_threshold=0.18,
        pickup_success_height=0.02,
        pickup_grasp_distance_threshold=0.18,
        pickup_progress_scale=3.0,
        pickup_dense_scale=0.35,
        press_success_threshold=0.1,
        press_progress_scale=6.0,
        press_dense_scale=0.25,
        toggle_progress_scale=0.5,
        toggle_progress_dense_scale=0.4,
        toggle_distance_dense_scale=0.2,
        placedown_success_height=0.12,
        placedown_progress_scale=3.0,
        placedown_dense_scale=0.25,
        toggle_success_reward=5.0,
        stage_completion_bonus=1.0,
        annotation_path=None,
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.inhand_infer_distance_threshold = inhand_infer_distance_threshold
        self.pickup_success_height = pickup_success_height
        self.pickup_grasp_distance_threshold = pickup_grasp_distance_threshold
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.press_success_threshold = press_success_threshold
        self.press_progress_scale = press_progress_scale
        self.press_dense_scale = press_dense_scale
        self.toggle_progress_scale = toggle_progress_scale
        self.toggle_progress_dense_scale = toggle_progress_dense_scale
        self.toggle_distance_dense_scale = toggle_distance_dense_scale
        self.placedown_success_height = placedown_success_height
        self.placedown_progress_scale = placedown_progress_scale
        self.placedown_dense_scale = placedown_dense_scale
        self.toggle_success_reward = toggle_success_reward
        self.annotation_path = annotation_path
        self._target_obj = None
        self._toggle_state = None
        self._support_obj = None
        self._support_surface_height = None
        self._initial_target_center_height = None
        self._support_label = None
        self._support_source = "unknown"
        self._has_left_support = False
        self._has_picked_up = False
        self._has_toggled_on = False
        self._toggle_steps_required = int(getattr(toggle_macros, "CAN_TOGGLE_STEPS", 5))
        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def reset(self, task, env):
        # Resolve task-specific objects once per episode. The reward logic assumes one toggleable target
        # and one support object that the radio is picked from and placed back onto.
        self._target_obj = self._find_toggleable_target(task)
        self._toggle_state = self._target_obj.states[ToggledOn] if self._target_obj is not None else None
        self._support_label = self._parse_support_label_from_annotation()
        self._support_obj = self._find_support_object(task)
        self._support_surface_height = self._infer_support_surface_height()
        self._initial_target_center_height = self._get_obj_center(self._target_obj)[2].item() if self._target_obj else None
        self._has_left_support = False
        self._has_picked_up = False
        self._has_toggled_on = False
        super().reset(task, env)

    def _find_toggleable_target(self, task):
        for obj in task.object_scope.values():
            if getattr(obj, "synset", None) == "agent":
                continue
            if hasattr(obj, "states") and ToggledOn in obj.states:
                return obj
        return None

    def _find_support_object(self, task):
        if self._target_obj is None:
            return None

        # First try to follow the human annotation ("coffee table"), which is much more stable than
        # reconstructing the support object from runtime states.
        if self._support_label is not None:
            support_obj = self._find_support_object_from_label(task, self._support_label)
            if support_obj is not None:
                self._support_source = "annotation"
                return support_obj

        # Fallback: recover the support object from the raw OnTop state when no annotation is available.
        for obj in task.object_scope.values():
            if obj == self._target_obj or getattr(obj, "synset", None) == "agent":
                continue
            try:
                if OnTop in self._target_obj.states and self._target_obj.states[OnTop].get_value(obj):
                    self._support_source = "ontop_fallback"
                    return obj
            except Exception:
                continue
        return None

    def _parse_support_label_from_annotation(self):
        if self.annotation_path is None or not os.path.exists(self.annotation_path):
            return None

        # We only need lightweight text parsing here: "pick up radio from coffee table" and
        # "place radio on coffee table" both expose the support object string directly.
        try:
            with open(self.annotation_path, "r") as f:
                data = json.load(f)
        except Exception:
            return None

        for subtask_text in data.get("cot_subtask_description_list", []):
            lowered = subtask_text.lower().strip()
            match = re.search(r"(?:pick up|pickup|place)\s+.+?\s+(?:from|on)\s+(.+)", lowered)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def _find_support_object_from_label(self, task, support_label):
        normalized_label = self._normalize_text(support_label)
        label_tokens = [tok for tok in normalized_label.split() if tok]
        best_obj = None
        best_score = -1

        for scope_name, obj in task.object_scope.items():
            if obj == self._target_obj or getattr(obj, "synset", None) == "agent":
                continue

            candidate_fields = [
                scope_name,
                getattr(obj, "name", ""),
                getattr(obj, "category", ""),
                getattr(obj, "model", ""),
                getattr(obj, "synset", ""),
            ]
            candidate_text = " ".join(self._normalize_text(field) for field in candidate_fields if field)
            score = sum(token in candidate_text for token in label_tokens)
            if normalized_label and normalized_label in candidate_text:
                score += 2

            if score > best_score:
                best_score = score
                best_obj = obj

        return best_obj if best_score > 0 else None

    def _infer_support_surface_height(self):
        if self._support_obj is None:
            return None
        center, _, extent, _ = self._support_obj.get_base_aligned_bbox(xy_aligned=True)
        return (center[2] + extent[2] / 2.0).item()

    def _build_stages(self, task, env):
        if self._target_obj is None or self._toggle_state is None:
            return [{"name": "missing_target"}]
        # Stage order is fixed and preserved in logs / video banner output.
        return [
            {"name": "move_to_radio", "state": {"prev_distance": None}},
            {"name": "pickup_from_support", "state": {"prev_eef_distance": None, "prev_height_gap": None}},
            {"name": "press_radio", "state": {"prev_distance": None, "prev_toggle_steps": None}},
            {"name": "place_on_support", "state": {"prev_eef_distance": None, "prev_height_gap": None}},
        ]

    def _get_robot(self, env):
        return env.robots[0]

    def _get_base_to_obj_distance(self, robot, obj):
        robot_xy = robot.get_position_orientation()[0][:2]
        center, _, extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
        half_extent_xy = extent[:2] / 2.0
        delta = th.abs(robot_xy - center[:2]) - half_extent_xy
        clamped = th.clamp(delta, min=0.0)
        return th.norm(clamped).item()

    def _get_obj_center(self, obj):
        return obj.get_position_orientation()[0]

    def _get_min_eef_distance_to_obj(self, robot, obj):
        obj_pos = self._get_obj_center(obj)
        dists = []
        for arm in getattr(robot, "arm_names", []):
            try:
                eef_pos = robot.get_eef_position(arm)
            except Exception:
                continue
            dists.append(th.norm(eef_pos - obj_pos).item())
        if not dists:
            default_arm = getattr(robot, "default_arm", None)
            if default_arm is not None:
                eef_pos = robot.get_eef_position(default_arm)
                dists.append(th.norm(eef_pos - obj_pos).item())
        return min(dists) if dists else float("inf")

    def _get_toggle_marker_position(self):
        if self._toggle_state is None or self._toggle_state.visual_marker is None:
            return self._target_obj.get_position_orientation()[0]
        return self._toggle_state.visual_marker.get_position_orientation()[0]

    def _get_toggle_marker_radius(self):
        if self._toggle_state is None or self._toggle_state.visual_marker is None:
            return 0.0
        return th.min(self._toggle_state.visual_marker.extent * self._toggle_state.scale).item()

    def _get_min_eef_distance_to_toggle(self, robot):
        toggle_pos = self._get_toggle_marker_position()
        dists = []
        for arm in getattr(robot, "arm_names", []):
            try:
                eef_pos = robot.get_eef_position(arm)
            except Exception:
                continue
            dists.append(th.norm(eef_pos - toggle_pos).item())
        if not dists:
            default_arm = getattr(robot, "default_arm", None)
            if default_arm is not None:
                eef_pos = robot.get_eef_position(default_arm)
                dists.append(th.norm(eef_pos - toggle_pos).item())
        return min(dists) if dists else float("inf")

    def _is_target_in_hand(self, robot):
        obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
        for arm in getattr(robot, "arm_names", []):
            if self._is_same_object(obj_in_hand.get(arm), self._target_obj):
                return True
        default_arm = getattr(robot, "default_arm", None)
        return default_arm is not None and self._is_same_object(obj_in_hand.get(default_arm), self._target_obj)

    @staticmethod
    def _is_same_object(obj_a, obj_b):
        if obj_a is None or obj_b is None:
            return False
        if obj_a is obj_b:
            return True

        # `_ag_obj_in_hand` sometimes returns an object handle that is semantically the same object but
        # not the exact same Python instance as `task.object_scope`. Compare several stable identifiers.
        comparable_attrs = ("prim_path", "name", "uuid")
        for attr in comparable_attrs:
            value_a = getattr(obj_a, attr, None)
            value_b = getattr(obj_b, attr, None)
            if value_a is not None and value_b is not None and value_a == value_b:
                return True
        return False

    def _get_in_hand_debug_info(self, robot):
        obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
        arm_names = list(getattr(robot, "arm_names", []))
        default_arm = getattr(robot, "default_arm", None)
        target_name = getattr(self._target_obj, "name", "None") if self._target_obj is not None else "None"
        target_prim_path = getattr(self._target_obj, "prim_path", "None") if self._target_obj is not None else "None"

        held_pairs = []
        strict_matches = []
        for arm in arm_names:
            held_obj = obj_in_hand.get(arm)
            held_name = getattr(held_obj, "name", "None") if held_obj is not None else "None"
            held_pairs.append(f"{arm}:{held_name}")
            strict_matches.append(f"{arm}:{self._is_same_object(held_obj, self._target_obj)}")

        if default_arm is not None and default_arm not in arm_names:
            held_obj = obj_in_hand.get(default_arm)
            held_name = getattr(held_obj, "name", "None") if held_obj is not None else "None"
            held_pairs.append(f"{default_arm}:{held_name}")
            strict_matches.append(f"{default_arm}:{self._is_same_object(held_obj, self._target_obj)}")

        return {
            "target_obj_name": target_name,
            "target_obj_prim_path": target_prim_path,
            "default_arm": str(default_arm),
            "held_objects_by_arm": "[" + ", ".join(held_pairs) + "]" if held_pairs else "[]",
            "strict_match_by_arm": "[" + ", ".join(strict_matches) + "]" if strict_matches else "[]",
        }

    def _is_target_in_hand_inferred(self, eef_distance):
        # This is intentionally weaker than the strict grasp state and is only used for shaping / release logic.
        return eef_distance <= self.inhand_infer_distance_threshold

    def _is_on_support(self):
        if self._support_obj is None or self._target_obj is None or OnTop not in self._target_obj.states:
            return False
        try:
            return bool(self._target_obj.states[OnTop].get_value(self._support_obj))
        except Exception:
            return False

    @staticmethod
    def _format_neighbor_names(neighbors):
        if not neighbors:
            return "[]"
        names = sorted(getattr(obj, "name", str(obj)) for obj in neighbors)
        return "[" + ", ".join(names) + "]"

    def _get_on_top_debug_info(self):
        default_info = {
            "touching_support": False,
            "support_in_negative_neighbors": False,
            "support_in_positive_neighbors": False,
            "vertical_negative_neighbors": "[]",
            "vertical_positive_neighbors": "[]",
        }
        if self._support_obj is None or self._target_obj is None:
            return default_info

        try:
            touching_support = bool(self._target_obj.states[Touching].get_value(self._support_obj))
        except Exception:
            touching_support = False

        try:
            adjacency = self._target_obj.states[VerticalAdjacency].get_value()
            negative_neighbors = getattr(adjacency, "negative_neighbors", set())
            positive_neighbors = getattr(adjacency, "positive_neighbors", set())
        except Exception:
            negative_neighbors = set()
            positive_neighbors = set()

        return {
            "touching_support": touching_support,
            "support_in_negative_neighbors": self._support_obj in negative_neighbors,
            "support_in_positive_neighbors": self._support_obj in positive_neighbors,
            "vertical_negative_neighbors": self._format_neighbor_names(negative_neighbors),
            "vertical_positive_neighbors": self._format_neighbor_names(positive_neighbors),
        }

    def _is_supported_by_surface(self):
        on_top_debug = self._get_on_top_debug_info()
        # Prefer the official OnTop state, but allow a fallback when Touching is noisy while the support
        # still appears below the object in vertical adjacency.
        supported = self._is_on_support() or on_top_debug["support_in_negative_neighbors"]
        return supported, on_top_debug

    @staticmethod
    def _is_task_bddl_success(task):
        try:
            # Reuse the same predicate termination status that drives task success in BehaviorTask.
            predicate_condition = task._termination_conditions.get("predicate", None)
            goal_status = None if predicate_condition is None else predicate_condition.goal_status
            return goal_status is not None and len(goal_status.get("unsatisfied", [])) == 0
        except Exception:
            return False

    def _get_height_above_support(self):
        if self._support_surface_height is None or self._target_obj is None:
            return 0.0
        obj_center_height = self._get_obj_center(self._target_obj)[2].item()
        return max(obj_center_height - self._support_surface_height, 0.0)

    def _get_height_above_initial(self):
        if self._initial_target_center_height is None or self._target_obj is None:
            return 0.0
        obj_center_height = self._get_obj_center(self._target_obj)[2].item()
        return max(obj_center_height - self._initial_target_center_height, 0.0)

    def _evaluate_stage(self, stage, task, env, action):
        if self._target_obj is None or self._toggle_state is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = self._get_robot(env)
        stage_state = stage["state"]
        stage_name = stage["name"]
        toggled_on = bool(self._toggle_state.get_value())
        self._has_toggled_on = self._has_toggled_on or toggled_on

        if stage_name == "move_to_radio":
            # Use nearest-EFF distance instead of base distance so the shaping generalizes better to
            # objects on tables, shelves, and the floor.
            distance = self._get_min_eef_distance_to_obj(robot, self._target_obj)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.move_to_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.move_to_dense_scale)
            stage_state["prev_distance"] = distance
            completed = distance <= self.move_to_success_threshold
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": distance,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "pickup_from_support":
            eef_distance = self._get_min_eef_distance_to_obj(robot, self._target_obj)
            height_gap = self._get_height_above_support()
            lift_from_initial = self._get_height_above_initial()
            target_center_height = self._get_obj_center(self._target_obj)[2].item()
            in_hand_strict = self._is_target_in_hand(robot)
            in_hand_inferred = self._is_target_in_hand_inferred(eef_distance)
            in_hand_debug = self._get_in_hand_debug_info(robot)
            on_support, on_top_debug = self._is_supported_by_surface()
            grasp_ready = in_hand_strict or in_hand_inferred or eef_distance <= self.pickup_grasp_distance_threshold
            lifted_off_support = (not on_support) and (
                height_gap >= self.pickup_success_height or lift_from_initial >= self.pickup_success_height
            )
            # Latch key transitions so a successful pickup is not "undone" just because the object is
            # later moved again while the policy proceeds to the next stage.
            self._has_left_support = self._has_left_support or lifted_off_support
            pickup_success_now = self._has_left_support and grasp_ready
            self._has_picked_up = self._has_picked_up or pickup_success_now or self._has_toggled_on
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], eef_distance, self.pickup_progress_scale, invert=True
            )
            lift_progress_reward = self._progress_reward(
                stage_state["prev_height_gap"], height_gap, self.pickup_progress_scale, invert=False
            )
            dense_reward = self._exp_distance_reward(eef_distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = eef_distance
            stage_state["prev_height_gap"] = height_gap
            completed = self._has_picked_up
            return {
                "reward": progress_reward + lift_progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": eef_distance,
                    "height_above_support": height_gap,
                    "height_above_initial": lift_from_initial,
                    "target_center_height": target_center_height,
                    "support_surface_height": self._support_surface_height if self._support_surface_height is not None else -1.0,
                    "support_obj_name": self._support_obj.name if self._support_obj is not None else "None",
                    "support_label": self._support_label or "None",
                    "support_source": self._support_source,
                    "annotation_path_exists": bool(self.annotation_path and os.path.exists(self.annotation_path)),
                    "on_support": on_support,
                    "ontop_state_raw": self._is_on_support(),
                    **on_top_debug,
                    "in_hand_strict": in_hand_strict,
                    "in_hand_inferred": in_hand_inferred,
                    **in_hand_debug,
                    "grasp_ready": grasp_ready,
                    "lifted_off_support": lifted_off_support,
                    "pickup_success_now": pickup_success_now,
                    "has_left_support": self._has_left_support,
                    "has_picked_up": self._has_picked_up,
                    "success_height": self.pickup_success_height,
                    "inhand_infer_distance_threshold": self.inhand_infer_distance_threshold,
                    "grasp_distance_threshold": self.pickup_grasp_distance_threshold,
                },
            }

        if stage_name == "press_radio":
            distance = self._get_min_eef_distance_to_toggle(robot)
            marker_radius = self._get_toggle_marker_radius()
            adjusted_distance = max(distance - marker_radius, 0.0)
            toggle_steps = int(self._toggle_state.robot_can_toggle_steps)
            target_center_height = self._get_obj_center(self._target_obj)[2].item()
            in_hand_strict = self._is_target_in_hand(robot)
            in_hand_inferred = self._is_target_in_hand_inferred(self._get_min_eef_distance_to_obj(robot, self._target_obj))
            in_hand_debug = self._get_in_hand_debug_info(robot)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], adjusted_distance, self.press_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(adjusted_distance, self.press_dense_scale)
            toggle_step_reward = self._progress_reward(
                stage_state["prev_toggle_steps"], toggle_steps, self.toggle_progress_scale, invert=False
            )
            toggle_progress_ratio = min(toggle_steps / max(self._toggle_steps_required, 1), 1.0)
            dense_progress_reward = toggle_progress_ratio * self.toggle_progress_dense_scale
            dense_distance_reward = self._exp_distance_reward(adjusted_distance, self.toggle_distance_dense_scale)
            stage_state["prev_distance"] = adjusted_distance
            stage_state["prev_toggle_steps"] = toggle_steps
            press_distance_ok = adjusted_distance <= self.press_success_threshold
            # Final success is governed by the real object state rather than distance alone.
            press_success_now = toggled_on
            completed = press_success_now
            return {
                "reward": progress_reward
                + dense_reward
                + toggle_step_reward
                + dense_progress_reward
                + dense_distance_reward
                + (self.toggle_success_reward if toggled_on else 0.0),
                "completed": completed,
                "metrics": {
                    "eef_to_toggle_distance": adjusted_distance,
                    "toggle_marker_radius": marker_radius,
                    "toggle_steps": toggle_steps,
                    "toggle_steps_required": self._toggle_steps_required,
                    "toggle_progress_ratio": toggle_progress_ratio,
                    "in_hand_strict": in_hand_strict,
                    "in_hand_inferred": in_hand_inferred,
                    **in_hand_debug,
                    "target_center_height": target_center_height,
                    "support_surface_height": self._support_surface_height if self._support_surface_height is not None else -1.0,
                    "support_obj_name": self._support_obj.name if self._support_obj is not None else "None",
                    "support_label": self._support_label or "None",
                    "support_source": self._support_source,
                    "annotation_path_exists": bool(self.annotation_path and os.path.exists(self.annotation_path)),
                    "toggled_on": toggled_on,
                    "press_distance_ok": press_distance_ok,
                    "press_success_now": press_success_now,
                    "success_threshold": self.press_success_threshold,
                },
            }

        if stage_name == "place_on_support":
            eef_distance = self._get_min_eef_distance_to_obj(robot, self._target_obj)
            height_gap = self._get_height_above_support()
            target_center_height = self._get_obj_center(self._target_obj)[2].item()
            in_hand_strict = self._is_target_in_hand(robot)
            in_hand_inferred = self._is_target_in_hand_inferred(eef_distance)
            in_hand_debug = self._get_in_hand_debug_info(robot)
            on_support, on_top_debug = self._is_supported_by_surface()
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], eef_distance, self.placedown_progress_scale, invert=True
            )
            settle_progress_reward = self._progress_reward(
                stage_state["prev_height_gap"], height_gap, self.placedown_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(eef_distance, self.placedown_dense_scale)
            stage_state["prev_eef_distance"] = eef_distance
            stage_state["prev_height_gap"] = height_gap
            place_height_ok = height_gap <= self.placedown_success_height
            # Release is intentionally weak: if the end-effector is still hugging the object, we keep
            # the stage active even if BDDL already says the task succeeded.
            released = not in_hand_inferred
            bddl_success = self._is_task_bddl_success(task)
            support_evidence = on_support or bddl_success
            # The final stage is allowed to trust BDDL success as a fallback because the annotation and
            # simulator truth already say the task is complete, while Touching / OnTop can be noisy.
            placedown_success_now = support_evidence and place_height_ok and released
            completed = placedown_success_now
            return {
                "reward": progress_reward + settle_progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": eef_distance,
                    "height_above_support": height_gap,
                    "target_center_height": target_center_height,
                    "support_surface_height": self._support_surface_height if self._support_surface_height is not None else -1.0,
                    "support_obj_name": self._support_obj.name if self._support_obj is not None else "None",
                    "support_label": self._support_label or "None",
                    "support_source": self._support_source,
                    "annotation_path_exists": bool(self.annotation_path and os.path.exists(self.annotation_path)),
                    "on_support": on_support,
                    "ontop_state_raw": self._is_on_support(),
                    **on_top_debug,
                    "in_hand_strict": in_hand_strict,
                    "in_hand_inferred": in_hand_inferred,
                    **in_hand_debug,
                    "place_height_ok": place_height_ok,
                    "released": released,
                    "bddl_success": bddl_success,
                    "support_evidence": support_evidence,
                    "placedown_success_now": placedown_success_now,
                    "success_height": self.placedown_success_height,
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {}}
