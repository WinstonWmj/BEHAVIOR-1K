from omnigibson.object_states.inside import Inside
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    get_min_eef_distance_to_obj,
    get_stage_objects_by_name,
    is_supported_by_surface,
    is_target_in_hand,
)


class PickingUpTrashReward(SequentialTaskReward):
    """Task-bound sequential reward for `picking_up_trash`."""

    STAGE_ANNOTATIONS = (
        {"skill_description": "move to", "object_id": ("trash_can_116",)},
        {"skill_description": "pick up from", "object_id": ("trash_can_116", "floors_zqjkvm_0")},
        {"skill_description": "move to", "object_id": ("can_of_soda_114",)},
        {"skill_description": "pick up from", "object_id": ("can_of_soda_114", "floors_ulujpr_0")},
        {"skill_description": "place in", "object_id": ("can_of_soda_114", "trash_can_116")},
        {"skill_description": "move to", "object_id": ("can_of_soda_115",)},
        {"skill_description": "pick up from", "object_id": ("can_of_soda_115", "floors_ulujpr_0")},
        {"skill_description": "place in", "object_id": ("can_of_soda_115", "trash_can_116")},
        {"skill_description": "move to", "object_id": ("can_of_soda_113",)},
        {"skill_description": "pick up from", "object_id": ("can_of_soda_113", "floors_ulujpr_0")},
        {"skill_description": "place in", "object_id": ("can_of_soda_113", "trash_can_116")},
        {"skill_description": "place on", "object_id": ("trash_can_116", "floors_ulujpr_0")},
    )

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=5.0,
        move_to_dense_scale=0.3,
        pickup_progress_scale=3.0,
        pickup_dense_scale=0.35,
        place_in_progress_scale=4.0,
        place_in_dense_scale=0.3,
        place_in_success_reward=5.0,
        place_on_progress_scale=3.0,
        place_on_dense_scale=0.25,
        stage_completion_bonus=1.0,
        reward_mode="task",
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.place_in_progress_scale = place_in_progress_scale
        self.place_in_dense_scale = place_in_dense_scale
        self.place_in_success_reward = place_in_success_reward
        self.place_on_progress_scale = place_on_progress_scale
        self.place_on_dense_scale = place_on_dense_scale

        self._stage_specs = []
        super().__init__(stage_completion_bonus=stage_completion_bonus, reward_mode=reward_mode)

    def reset(self, task, env):
        stage_annotations = self.STAGE_ANNOTATIONS
        self._stage_specs = []

        for stage_idx, stage_annotation in enumerate(stage_annotations):
            stage_objects = get_stage_objects_by_name(env, stage_annotation.get("object_id", []))
            object_names = list(stage_annotation.get("object_id", []))
            skill = (stage_annotation.get("skill_description") or "").strip().lower()

            target_obj = stage_objects[0] if len(stage_objects) > 0 else None
            secondary_obj = stage_objects[1] if len(stage_objects) > 1 else None
            target_name = object_names[0] if len(object_names) > 0 else f"obj_{stage_idx}"

            if skill == "move to":
                stage_name = f"move_to_{target_name}"
                stage_type = "move_to"
            elif skill == "pick up from":
                stage_name = f"pickup_{target_name}_from_{object_names[1] if len(object_names) > 1 else 'support'}"
                stage_type = "pick_up_from"
            elif skill == "place in":
                stage_name = f"place_{target_name}_in_{object_names[1] if len(object_names) > 1 else 'container'}"
                stage_type = "place_in"
            elif skill == "place on":
                stage_name = f"place_{target_name}_on_{object_names[1] if len(object_names) > 1 else 'support'}"
                stage_type = "place_on"
            else:
                stage_name = f"stage_{stage_idx}_{skill.replace(' ', '_') or 'unknown'}"
                stage_type = "unknown"

            self._stage_specs.append(
                {
                    "name": stage_name,
                    "stage_type": stage_type,
                    "objects": stage_objects,
                    "object_names": object_names,
                    "annotation": stage_annotation,
                    "target_obj": target_obj,
                    "secondary_obj": secondary_obj,
                }
            )

        super().reset(task, env)

    def _build_stages(self, task, env):
        if not self._stage_specs:
            return [{"name": "missing_target", "stage_type": "missing_target", "state": {}}]

        stages = []
        # each stage has their own local state;
        for spec in self._stage_specs:
            stage = {
                "name": spec["name"],
                "stage_type": spec["stage_type"],
                "objects": spec["objects"],
                "target_obj": spec["target_obj"],
                "secondary_obj": spec["secondary_obj"],
            }

            if spec["stage_type"] == "move_to":
                stage["state"] = {"prev_distance": None}
            elif spec["stage_type"] == "pick_up_from":
                stage["state"] = {
                    "prev_eef_distance": None,
                    "has_left_support": False,
                    "has_picked_up": False,
                }
            elif spec["stage_type"] == "place_in":
                stage["state"] = {"prev_distance": None, "was_inside": False}
            elif spec["stage_type"] == "place_on":
                stage["state"] = {"prev_eef_distance": None}
            else:
                stage["state"] = {}

            stages.append(stage)

        return stages

    def _evaluate_stage(self, stage, task, env, action):
        del task, action
        if stage["stage_type"] == "missing_target":
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = env.robots[0]
        stage_state = stage["state"]
        stage_type = stage["stage_type"]
        target_obj = stage["target_obj"]
        secondary_obj = stage["secondary_obj"]

        if target_obj is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        in_hand = is_target_in_hand(robot, target_obj)

        if stage_type == "move_to":
            distance = get_min_eef_distance_to_obj(robot, target_obj)
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
                    "eef_to_target_distance": distance,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_type == "pick_up_from":
            distance = get_min_eef_distance_to_obj(robot, target_obj)
            on_support = is_supported_by_surface(target_obj, secondary_obj)
            stage_state["has_left_support"] = stage_state["has_left_support"] or (not on_support)
            stage_state["has_picked_up"] = stage_state["has_picked_up"] or (
                stage_state["has_left_support"] and in_hand
            )
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.pickup_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = distance
            return {
                "reward": progress_reward + dense_reward,
                "completed": stage_state["has_picked_up"],
                "metrics": {
                    "eef_to_target_distance": distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                    "has_left_support": stage_state["has_left_support"],
                    "has_picked_up": stage_state["has_picked_up"],
                },
            }

        if stage_type == "place_in":
            if secondary_obj is None:
                return {"reward": 0.0, "completed": False, "metrics": {"missing_container": True}}

            distance = get_min_eef_distance_to_obj(robot, secondary_obj)
            inside_container = Inside in target_obj.states and bool(target_obj.states[Inside].get_value(secondary_obj))
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.place_in_progress_scale, invert=True
            )
            if inside_container and not stage_state["was_inside"]:
                progress_reward += self.place_in_progress_scale
            dense_reward = self._exp_distance_reward(distance, self.place_in_dense_scale)
            stage_state["prev_distance"] = distance
            stage_state["was_inside"] = inside_container
            reward = progress_reward + dense_reward
            if inside_container:
                reward += self.place_in_success_reward
            return {
                "reward": reward,
                "completed": inside_container and (not in_hand),
                "metrics": {
                    "eef_to_container_distance": distance,
                    "inside_container": inside_container,
                    "in_hand": in_hand,
                },
            }

        if stage_type == "place_on":
            distance = get_min_eef_distance_to_obj(robot, target_obj)
            on_support = is_supported_by_surface(target_obj, secondary_obj)
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.place_on_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.place_on_dense_scale)
            stage_state["prev_eef_distance"] = distance
            return {
                "reward": progress_reward + dense_reward,
                "completed": on_support and (not in_hand),
                "metrics": {
                    "eef_to_target_distance": distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {"unsupported_stage_type": stage_type}}
