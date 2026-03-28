import json
import logging
import math
import os
import re

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.attached_to import AttachedTo
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.touching import Touching
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward

log = logging.getLogger("evaluator")


class HangingPicturesReward(SequentialTaskReward):
    """
    Task-bound sequential reward for `hanging_pictures`.

    Stages:
    1. Move close to the poster.
    2. Pick the poster up from the bar.
    3. Move the poster close to the target wall nail.
    4. Align and attach the poster to the wall nail.
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
        pickup_confirmation_steps=3,
        move_to_hang_success_threshold=0.14,
        move_to_hang_progress_scale=6.0,
        move_to_hang_dense_scale=0.3,
        hang_position_progress_scale=8.0,
        hang_orientation_progress_scale=1.5,
        hang_position_dense_scale=0.35,
        hang_orientation_dense_scale=0.15,
        hang_grasp_dense_scale=0.1,
        hang_success_reward=5.0,
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
        self.pickup_confirmation_steps = pickup_confirmation_steps
        self.move_to_hang_success_threshold = move_to_hang_success_threshold
        self.move_to_hang_progress_scale = move_to_hang_progress_scale
        self.move_to_hang_dense_scale = move_to_hang_dense_scale
        self.hang_position_progress_scale = hang_position_progress_scale
        self.hang_orientation_progress_scale = hang_orientation_progress_scale
        self.hang_position_dense_scale = hang_position_dense_scale
        self.hang_orientation_dense_scale = hang_orientation_dense_scale
        self.hang_grasp_dense_scale = hang_grasp_dense_scale
        self.hang_success_reward = hang_success_reward
        self.annotation_path = annotation_path

        self._poster_obj = None
        self._bar_obj = None
        self._wall_nail_obj = None
        self._poster_label = None
        self._bar_label = None
        self._wall_nail_label = None
        self._bar_surface_height = None
        self._initial_poster_center_height = None
        self._has_left_bar = False
        self._has_picked_up = False
        self._pickup_success_streak = 0

        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def reset(self, task, env):
        parsed_ids = self._parse_annotation_ids()
        self._poster_label = parsed_ids.get("poster")
        self._bar_label = parsed_ids.get("bar")
        self._wall_nail_label = parsed_ids.get("wall_nail")

        self._poster_obj = self._find_object(task, preferred_label=self._poster_label, preferred_category="poster")
        self._bar_obj = self._find_object(task, preferred_label=self._bar_label, preferred_category="bar")
        self._wall_nail_obj = self._find_object(
            task, preferred_label=self._wall_nail_label, preferred_category="wall_nail"
        )
        self._bar_surface_height = self._infer_surface_height(self._bar_obj)
        self._initial_poster_center_height = self._get_obj_center(self._poster_obj)[2].item() if self._poster_obj else None
        self._has_left_bar = False
        self._has_picked_up = False
        self._pickup_success_streak = 0
        log.info(
            "HangingPicturesReward reset: annotation_exists=%s poster=%s bar=%s wall_nail=%s",
            bool(self.annotation_path and os.path.exists(self.annotation_path)),
            self._safe_name(self._poster_obj),
            self._safe_name(self._bar_obj),
            self._safe_name(self._wall_nail_obj),
        )
        super().reset(task, env)

    def _build_stages(self, task, env):
        if self._poster_obj is None:
            return [{"name": "missing_target"}]

        return [
            {"name": "move_to_poster", "state": {"prev_distance": None}},
            {"name": "pickup_from_bar", "state": {"prev_eef_distance": None, "prev_height_gap": None}},
            {"name": "move_to_wall_nail", "state": {"prev_distance": None}},
            {
                "name": "hang_on_wall_nail",
                "state": {
                    "prev_attach_distance": None,
                    "prev_attach_orientation": None,
                    "prev_eef_distance": None,
                },
            },
        ]

    def _evaluate_stage(self, stage, task, env, action):
        if self._poster_obj is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = self._get_robot(env)
        stage_state = stage["state"]
        stage_name = stage["name"]

        if stage_name == "move_to_poster":
            distance = self._get_min_eef_distance_to_obj(robot, self._poster_obj)
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
                    "eef_to_poster_distance": distance,
                    "poster_obj_name": self._safe_name(self._poster_obj),
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "pickup_from_bar":
            eef_distance = self._get_min_eef_distance_to_obj(robot, self._poster_obj)
            height_gap = self._get_height_above_bar()
            lift_from_initial = self._get_height_above_initial()
            in_hand_strict = self._is_target_in_hand(robot, self._poster_obj)
            in_hand_inferred = self._is_target_in_hand_inferred(eef_distance)
            on_bar, on_top_debug = self._is_supported_by_surface(self._poster_obj, self._bar_obj)
            support_ready = self._bar_obj is not None and self._bar_surface_height is not None
            grasp_ready = in_hand_strict or in_hand_inferred
            lifted_off_bar = support_ready and (not on_bar) and (
                height_gap >= self.pickup_success_height or lift_from_initial >= self.pickup_success_height
            )
            self._has_left_bar = self._has_left_bar or lifted_off_bar
            pickup_candidate = support_ready and lifted_off_bar and grasp_ready
            self._pickup_success_streak = self._pickup_success_streak + 1 if pickup_candidate else 0
            pickup_success_now = self._pickup_success_streak >= self.pickup_confirmation_steps
            self._has_picked_up = self._has_picked_up or pickup_success_now

            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], eef_distance, self.pickup_progress_scale, invert=True
            )
            lift_progress_reward = self._progress_reward(
                stage_state["prev_height_gap"], height_gap, self.pickup_progress_scale, invert=False
            )
            dense_reward = self._exp_distance_reward(eef_distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = eef_distance
            stage_state["prev_height_gap"] = height_gap

            return {
                "reward": progress_reward + lift_progress_reward + dense_reward,
                "completed": self._has_picked_up,
                "metrics": {
                    "eef_to_poster_distance": eef_distance,
                    "height_above_bar": height_gap,
                    "height_above_initial": lift_from_initial,
                    "poster_obj_name": self._safe_name(self._poster_obj),
                    "bar_obj_name": self._safe_name(self._bar_obj),
                    "bar_surface_height": self._bar_surface_height if self._bar_surface_height is not None else -1.0,
                    "support_ready": support_ready,
                    "on_bar": on_bar,
                    "ontop_state_raw": self._is_on_support(self._poster_obj, self._bar_obj),
                    **on_top_debug,
                    "in_hand_strict": in_hand_strict,
                    "in_hand_inferred": in_hand_inferred,
                    "grasp_ready": grasp_ready,
                    "lifted_off_bar": lifted_off_bar,
                    "pickup_candidate": pickup_candidate,
                    "pickup_success_streak": self._pickup_success_streak,
                    "pickup_success_now": pickup_success_now,
                    "has_left_bar": self._has_left_bar,
                    "has_picked_up": self._has_picked_up,
                    "annotation_path_exists": bool(self.annotation_path and os.path.exists(self.annotation_path)),
                    "pickup_confirmation_steps": self.pickup_confirmation_steps,
                },
            }

        if stage_name == "move_to_wall_nail":
            attach_distance, attach_orientation, has_candidate = self._get_attachment_alignment_errors()
            if has_candidate:
                distance = attach_distance
            else:
                distance = self._get_min_eef_distance_to_obj(robot, self._wall_nail_obj)
            if math.isfinite(distance):
                progress_reward = self._progress_reward(
                    stage_state["prev_distance"], distance, self.move_to_hang_progress_scale, invert=True
                )
                dense_reward = self._exp_distance_reward(distance, self.move_to_hang_dense_scale)
                stage_state["prev_distance"] = distance
            else:
                progress_reward = 0.0
                dense_reward = 0.0
                stage_state["prev_distance"] = None
            completed = distance <= self.move_to_hang_success_threshold
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "poster_obj_name": self._safe_name(self._poster_obj),
                    "wall_nail_obj_name": self._safe_name(self._wall_nail_obj),
                    "attach_distance": attach_distance,
                    "attach_orientation_error": attach_orientation,
                    "has_attachment_candidate": has_candidate,
                    "success_threshold": self.move_to_hang_success_threshold,
                },
            }

        if stage_name == "hang_on_wall_nail":
            attach_distance, attach_orientation, has_candidate = self._get_attachment_alignment_errors()
            eef_distance = self._get_min_eef_distance_to_obj(robot, self._poster_obj)
            attached = self._is_attached_to_target()
            in_hand_strict = self._is_target_in_hand(robot, self._poster_obj)
            in_hand_inferred = self._is_target_in_hand_inferred(eef_distance)

            pos_progress_reward = self._progress_reward(
                stage_state["prev_attach_distance"],
                attach_distance,
                self.hang_position_progress_scale,
                invert=True,
            )
            orn_progress_reward = self._progress_reward(
                stage_state["prev_attach_orientation"],
                attach_orientation,
                self.hang_orientation_progress_scale,
                invert=True,
            )
            pose_dense_reward = self._exp_distance_reward(attach_distance, self.hang_position_dense_scale)
            orn_dense_reward = math.exp(-max(attach_orientation, 0.0)) * self.hang_orientation_dense_scale
            grasp_dense_reward = self._exp_distance_reward(eef_distance, self.hang_grasp_dense_scale)

            stage_state["prev_attach_distance"] = attach_distance
            stage_state["prev_attach_orientation"] = attach_orientation
            stage_state["prev_eef_distance"] = eef_distance

            reward = pos_progress_reward + orn_progress_reward + pose_dense_reward + orn_dense_reward + grasp_dense_reward
            if attached:
                reward += self.hang_success_reward

            return {
                "reward": reward,
                "completed": attached,
                "metrics": {
                    "poster_obj_name": self._safe_name(self._poster_obj),
                    "wall_nail_obj_name": self._safe_name(self._wall_nail_obj),
                    "attach_distance": attach_distance,
                    "attach_orientation_error": attach_orientation,
                    "eef_to_poster_distance": eef_distance,
                    "has_attachment_candidate": has_candidate,
                    "attached_to_wall_nail": attached,
                    "in_hand_strict": in_hand_strict,
                    "in_hand_inferred": in_hand_inferred,
                    "annotation_path_exists": bool(self.annotation_path and os.path.exists(self.annotation_path)),
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {}}

    def _parse_annotation_ids(self):
        ids = {"poster": None, "bar": None, "wall_nail": None}
        if self.annotation_path is None or not os.path.exists(self.annotation_path):
            return ids

        try:
            with open(self.annotation_path, "r") as f:
                data = json.load(f)
        except Exception:
            return ids

        skill_annotations = data.get("skill_annotation", [])
        for skill in skill_annotations:
            descriptions = [str(desc).lower() for desc in skill.get("skill_description", [])]
            object_groups = skill.get("object_id", [])
            if not object_groups:
                continue
            flat_ids = [obj_id for group in object_groups for obj_id in group]
            if any("pick up" in desc for desc in descriptions):
                ids["poster"] = ids["poster"] or self._first_matching_id(flat_ids, "poster")
                ids["bar"] = ids["bar"] or self._first_matching_id(flat_ids, "bar")
            if any("hang" in desc for desc in descriptions):
                ids["poster"] = ids["poster"] or self._first_matching_id(flat_ids, "poster")
                ids["wall_nail"] = ids["wall_nail"] or self._first_matching_id(flat_ids, "wall_nail")
            if any("move to" in desc for desc in descriptions):
                ids["poster"] = ids["poster"] or self._first_matching_id(flat_ids, "poster")
                ids["wall_nail"] = ids["wall_nail"] or self._first_matching_id(flat_ids, "wall_nail")

        if any(value is None for value in ids.values()):
            for subtask_text in data.get("cot_subtask_description_list", []):
                lowered = str(subtask_text).lower().strip()
                if "pick up" in lowered and " from " in lowered:
                    ids["bar"] = ids["bar"] or lowered.split(" from ", 1)[1].strip()
                if "hang" in lowered and " on " in lowered:
                    ids["wall_nail"] = ids["wall_nail"] or lowered.split(" on ", 1)[1].strip()
                if "poster" in lowered:
                    ids["poster"] = ids["poster"] or "poster"

        return ids

    @staticmethod
    def _first_matching_id(candidates, keyword):
        for candidate in candidates:
            if keyword in str(candidate).lower():
                return candidate
        return None

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def _find_object(self, task, preferred_label=None, preferred_category=None):
        normalized_label = self._normalize_text(preferred_label) if preferred_label else None
        label_tokens = [tok for tok in (normalized_label or "").split() if tok]
        best_obj = None
        best_score = -1

        for scope_name, obj in task.object_scope.items():
            if obj is None or getattr(obj, "synset", None) == "agent":
                continue

            candidate_fields = [
                scope_name,
                getattr(obj, "name", ""),
                getattr(obj, "category", ""),
                getattr(obj, "model", ""),
                getattr(obj, "synset", ""),
            ]
            candidate_text = " ".join(self._normalize_text(field) for field in candidate_fields if field)
            score = 0

            if normalized_label:
                score += 3 if normalized_label in candidate_text else 0
                score += sum(token in candidate_text for token in label_tokens)
            if preferred_category:
                score += 2 if preferred_category in candidate_text else 0

            if score > best_score:
                best_score = score
                best_obj = obj

        return best_obj if best_score > 0 else None

    def _get_robot(self, env):
        return env.robots[0]

    @staticmethod
    def _safe_name(obj):
        return getattr(obj, "name", "None") if obj is not None else "None"

    def _get_obj_center(self, obj):
        return obj.get_position_orientation()[0]

    def _get_min_eef_distance_to_obj(self, robot, obj):
        if obj is None:
            return float("inf")
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

    def _is_target_in_hand(self, robot, target_obj):
        obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
        for arm in getattr(robot, "arm_names", []):
            if self._is_same_object(obj_in_hand.get(arm), target_obj):
                return True
        default_arm = getattr(robot, "default_arm", None)
        return default_arm is not None and self._is_same_object(obj_in_hand.get(default_arm), target_obj)

    @staticmethod
    def _is_same_object(obj_a, obj_b):
        if obj_a is None or obj_b is None:
            return False
        if obj_a is obj_b:
            return True

        comparable_attrs = ("prim_path", "name", "uuid")
        for attr in comparable_attrs:
            value_a = getattr(obj_a, attr, None)
            value_b = getattr(obj_b, attr, None)
            if value_a is not None and value_b is not None and value_a == value_b:
                return True
        return False

    def _is_target_in_hand_inferred(self, eef_distance):
        return eef_distance <= self.inhand_infer_distance_threshold

    @staticmethod
    def _format_neighbor_names(neighbors):
        if not neighbors:
            return "[]"
        names = sorted(getattr(obj, "name", str(obj)) for obj in neighbors)
        return "[" + ", ".join(names) + "]"

    def _is_on_support(self, child_obj, support_obj):
        if child_obj is None or support_obj is None or OnTop not in child_obj.states:
            return False
        try:
            return bool(child_obj.states[OnTop].get_value(support_obj))
        except Exception:
            return False

    def _is_supported_by_surface(self, child_obj, support_obj):
        default_info = {
            "touching_support": False,
            "support_in_negative_neighbors": False,
            "support_in_positive_neighbors": False,
            "vertical_negative_neighbors": "[]",
            "vertical_positive_neighbors": "[]",
        }
        if child_obj is None or support_obj is None:
            return False, default_info

        try:
            touching_support = bool(child_obj.states[Touching].get_value(support_obj))
        except Exception:
            touching_support = False

        try:
            adjacency = child_obj.states[VerticalAdjacency].get_value()
            negative_neighbors = getattr(adjacency, "negative_neighbors", set())
            positive_neighbors = getattr(adjacency, "positive_neighbors", set())
        except Exception:
            negative_neighbors = set()
            positive_neighbors = set()

        on_top = self._is_on_support(child_obj, support_obj)
        debug = {
            "touching_support": touching_support,
            "support_in_negative_neighbors": support_obj in negative_neighbors,
            "support_in_positive_neighbors": support_obj in positive_neighbors,
            "vertical_negative_neighbors": self._format_neighbor_names(negative_neighbors),
            "vertical_positive_neighbors": self._format_neighbor_names(positive_neighbors),
        }
        return on_top or debug["support_in_negative_neighbors"], debug

    @staticmethod
    def _infer_surface_height(obj):
        if obj is None:
            return None
        center, _, extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
        return (center[2] + extent[2] / 2.0).item()

    def _get_height_above_bar(self):
        if self._bar_surface_height is None or self._poster_obj is None:
            return 0.0
        poster_center_height = self._get_obj_center(self._poster_obj)[2].item()
        return max(poster_center_height - self._bar_surface_height, 0.0)

    def _get_height_above_initial(self):
        if self._initial_poster_center_height is None or self._poster_obj is None:
            return 0.0
        poster_center_height = self._get_obj_center(self._poster_obj)[2].item()
        return max(poster_center_height - self._initial_poster_center_height, 0.0)

    def _is_attached_to_target(self):
        if self._poster_obj is None or self._wall_nail_obj is None or AttachedTo not in self._poster_obj.states:
            return False
        try:
            return bool(self._poster_obj.states[AttachedTo].get_value(self._wall_nail_obj))
        except Exception:
            return False

    def _get_attachment_alignment_errors(self):
        if self._poster_obj is None or self._wall_nail_obj is None or AttachedTo not in self._poster_obj.states:
            return float("inf"), math.pi, False

        try:
            candidates = self._poster_obj.states[AttachedTo]._get_parent_candidates(self._wall_nail_obj)
        except Exception:
            candidates = None

        if not candidates:
            return float("inf"), math.pi, False

        best_distance = float("inf")
        best_orientation = math.pi
        has_candidate = False

        for child_link_name, parent_link_names in candidates.items():
            child_link = self._poster_obj.states[AttachedTo].links[child_link_name]
            child_pos, child_quat = child_link.get_position_orientation()
            for parent_link_name in parent_link_names:
                parent_link = self._wall_nail_obj.states[AttachedTo].links[parent_link_name]
                parent_pos, parent_quat = parent_link.get_position_orientation()
                pos_diff = th.norm(child_pos - parent_pos).item()
                orn_diff = T.get_orientation_diff_in_radian(child_quat, parent_quat)
                if pos_diff < best_distance or (math.isclose(pos_diff, best_distance) and orn_diff < best_orientation):
                    best_distance = pos_diff
                    best_orientation = float(orn_diff)
                    has_candidate = True

        return best_distance, best_orientation, has_candidate
