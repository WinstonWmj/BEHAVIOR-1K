import math

from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    load_orchestrator_stage_annotations,
    get_stage_objects,
    get_attachment_alignment_errors,
    get_min_eef_distance_to_obj,
    is_attached_to_target,
    is_supported_by_surface,
    is_target_in_hand,
)


class HangingPicturesReward(SequentialTaskReward):
    """Task-bound sequential reward for `hanging_pictures`."""

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=5.0,
        move_to_dense_scale=0.3,
        inhand_infer_distance_threshold=0.18,
        pickup_progress_scale=3.0,
        pickup_dense_scale=0.35,
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
        orchestrators_annotation_dir=None,
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.inhand_infer_distance_threshold = inhand_infer_distance_threshold
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.move_to_hang_success_threshold = move_to_hang_success_threshold
        self.move_to_hang_progress_scale = move_to_hang_progress_scale
        self.move_to_hang_dense_scale = move_to_hang_dense_scale
        self.hang_position_progress_scale = hang_position_progress_scale
        self.hang_orientation_progress_scale = hang_orientation_progress_scale
        self.hang_position_dense_scale = hang_position_dense_scale
        self.hang_orientation_dense_scale = hang_orientation_dense_scale
        self.hang_grasp_dense_scale = hang_grasp_dense_scale
        self.hang_success_reward = hang_success_reward
        self.orchestrators_annotation_dir = orchestrators_annotation_dir

        self._poster_obj = None
        self._support_obj = None
        self._wall_nail_obj = None
        self._stage_objects = {}
        self._has_left_support = False
        self._has_picked_up = False
        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def reset(self, task, env):
        stage_annotations = load_orchestrator_stage_annotations(self.orchestrators_annotation_dir)
        self._stage_objects = {
            "move_to_poster": get_stage_objects(env, stage_annotations[0]),
            "pickup_from_bar": get_stage_objects(env, stage_annotations[1]),
            "move_to_wall_nail": get_stage_objects(env, stage_annotations[2]),
            "hang_on_wall_nail": get_stage_objects(env, stage_annotations[3]),
        }
        self._poster_obj = self._stage_objects["move_to_poster"][0] if self._stage_objects["move_to_poster"] else None
        self._support_obj = self._stage_objects["pickup_from_bar"][1] if len(self._stage_objects["pickup_from_bar"]) > 1 else None
        self._wall_nail_obj = self._stage_objects["move_to_wall_nail"][0] if self._stage_objects["move_to_wall_nail"] else None
        self._has_left_support = False
        self._has_picked_up = False
        super().reset(task, env)

    def _build_stages(self, task, env):
        if self._poster_obj is None:
            return [{"name": "missing_target"}]

        return [
            {
                "name": "move_to_poster",
                "objects": self._stage_objects.get("move_to_poster", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "pickup_from_bar",
                "objects": self._stage_objects.get("pickup_from_bar", []),
                "state": {"prev_eef_distance": None},
            },
            {
                "name": "move_to_wall_nail",
                "objects": self._stage_objects.get("move_to_wall_nail", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "hang_on_wall_nail",
                "objects": self._stage_objects.get("hang_on_wall_nail", []),
                "state": {
                    "prev_attach_distance": None,
                    "prev_attach_orientation": None,
                    "prev_eef_distance": None,
                },
            },
        ]

    def _evaluate_stage(self, stage, task, env, action):
        del task, action
        if self._poster_obj is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = env.robots[0]
        stage_state = stage["state"]
        stage_name = stage["name"]

        if stage_name == "move_to_poster":
            distance = get_min_eef_distance_to_obj(robot, self._poster_obj)
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
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "pickup_from_bar":
            eef_distance = get_min_eef_distance_to_obj(robot, self._poster_obj)
            in_hand = is_target_in_hand(robot, self._poster_obj)
            on_support = is_supported_by_surface(
                self._poster_obj,
                self._support_obj,
            )
            self._has_left_support = self._has_left_support or (not on_support)
            self._has_picked_up = self._has_picked_up or (self._has_left_support and in_hand)

            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], eef_distance, self.pickup_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(eef_distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = eef_distance
            return {
                "reward": progress_reward + dense_reward,
                "completed": self._has_picked_up,
                "metrics": {
                    "eef_to_poster_distance": eef_distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                    "has_left_support": self._has_left_support,
                    "has_picked_up": self._has_picked_up,
                },
            }

        if stage_name == "move_to_wall_nail":
            attach_distance, attach_orientation, has_candidate = get_attachment_alignment_errors(
                self._poster_obj, self._wall_nail_obj
            )
            distance = attach_distance if has_candidate else get_min_eef_distance_to_obj(robot, self._wall_nail_obj)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.move_to_hang_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.move_to_hang_dense_scale)
            stage_state["prev_distance"] = distance
            completed = distance <= self.move_to_hang_success_threshold
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "attach_distance": attach_distance,
                    "attach_orientation_error": attach_orientation,
                    "has_attachment_candidate": has_candidate,
                    "success_threshold": self.move_to_hang_success_threshold,
                },
            }

        if stage_name == "hang_on_wall_nail":
            attach_distance, attach_orientation, has_candidate = get_attachment_alignment_errors(
                self._poster_obj, self._wall_nail_obj
            )
            eef_distance = get_min_eef_distance_to_obj(robot, self._poster_obj)
            attached = is_attached_to_target(self._poster_obj, self._wall_nail_obj)
            in_hand = is_target_in_hand(robot, self._poster_obj) or (
                eef_distance <= self.inhand_infer_distance_threshold
            )

            progress_reward = self._progress_reward(
                stage_state["prev_attach_distance"],
                attach_distance,
                self.hang_position_progress_scale,
                invert=True,
            ) + self._progress_reward(
                stage_state["prev_attach_orientation"],
                attach_orientation,
                self.hang_orientation_progress_scale,
                invert=True,
            )
            dense_reward = self._exp_distance_reward(attach_distance, self.hang_position_dense_scale) + (
                math.exp(-max(attach_orientation, 0.0)) * self.hang_orientation_dense_scale
            )
            if in_hand:
                dense_reward += self._exp_distance_reward(eef_distance, self.hang_grasp_dense_scale)

            stage_state["prev_attach_distance"] = attach_distance
            stage_state["prev_attach_orientation"] = attach_orientation
            stage_state["prev_eef_distance"] = eef_distance
            reward = progress_reward + dense_reward
            if attached:
                reward += self.hang_success_reward

            return {
                "reward": reward,
                "completed": attached,
                "metrics": {
                    "attach_distance": attach_distance,
                    "attach_orientation_error": attach_orientation,
                    "eef_to_poster_distance": eef_distance,
                    "has_attachment_candidate": has_candidate,
                    "attached_to_wall_nail": attached,
                    "in_hand": in_hand,
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {}}
