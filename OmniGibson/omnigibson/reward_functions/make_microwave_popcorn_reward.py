from omnigibson.object_states.inside import Inside
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.toggle import ToggledOn, m as toggle_macros
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    get_min_eef_distance_to_obj,
    get_min_eef_distance_to_toggle,
    get_stage_objects,
    is_supported_by_surface,
    is_target_in_hand,
    load_orchestrator_stage_annotations,
)


class MakeMicrowavePopcornReward(SequentialTaskReward):
    """Task-bound sequential reward for `make_microwave_popcorn`."""

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=5.0,
        move_to_dense_scale=0.3,
        microwave_open_ratio_threshold=0.8,
        open_progress_scale=4.0,
        open_dense_scale=0.25,
        pickup_progress_scale=3.0,
        pickup_dense_scale=0.35,
        return_to_microwave_progress_scale=5.0,
        return_to_microwave_dense_scale=0.3,
        place_in_progress_scale=4.0,
        place_in_dense_scale=0.3,
        close_progress_scale=4.0,
        close_dense_scale=0.25,
        press_progress_scale=6.0,
        press_dense_scale=0.25,
        toggle_progress_scale=0.5,
        toggle_progress_dense_scale=0.4,
        stage_completion_bonus=1.0,
        orchestrators_annotation_dir=None,
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.microwave_open_ratio_threshold = microwave_open_ratio_threshold
        self.open_progress_scale = open_progress_scale
        self.open_dense_scale = open_dense_scale
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.return_to_microwave_progress_scale = return_to_microwave_progress_scale
        self.return_to_microwave_dense_scale = return_to_microwave_dense_scale
        self.place_in_progress_scale = place_in_progress_scale
        self.place_in_dense_scale = place_in_dense_scale
        self.close_progress_scale = close_progress_scale
        self.close_dense_scale = close_dense_scale
        self.press_progress_scale = press_progress_scale
        self.press_dense_scale = press_dense_scale
        self.toggle_progress_scale = toggle_progress_scale
        self.toggle_progress_dense_scale = toggle_progress_dense_scale
        self.orchestrators_annotation_dir = orchestrators_annotation_dir

        self._microwave_obj = None
        self._popcorn_obj = None
        self._support_obj = None
        self._open_state = None
        self._toggle_state = None
        self._stage_objects = {}
        self._has_left_support = False
        self._has_picked_up = False
        self._toggle_steps_required = int(getattr(toggle_macros, "CAN_TOGGLE_STEPS", 5))
        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def _is_microwave_open(self):
        both_sides, relevant_joints, joint_directions = self._open_state.relevant_joints_info
        if not relevant_joints:
            return False

        sides = [1, -1] if both_sides else [1]
        sides_openness = []
        for side in sides:
            joint_openness = []
            for joint, joint_direction in zip(relevant_joints, joint_directions):
                closed_end = joint.lower_limit if joint_direction * side == 1 else joint.upper_limit
                open_end = joint.upper_limit if joint_direction * side == 1 else joint.lower_limit
                joint_range = abs(open_end - closed_end)
                position = joint.get_state()[0]
                if joint_range <= 1e-6:
                    openness_ratio = 0.0
                else:
                    openness_ratio = abs(position - closed_end) / joint_range
                joint_openness.append(openness_ratio >= self.microwave_open_ratio_threshold)
            sides_openness.append(any(joint_openness))

        return all(sides_openness)

    def reset(self, task, env):
        stage_annotations = load_orchestrator_stage_annotations(self.orchestrators_annotation_dir)
        self._stage_objects = {
            "move_to_microwave": get_stage_objects(env, stage_annotations[0]),
            "open_microwave_door": get_stage_objects(env, stage_annotations[1]),
            "move_to_popcorn": get_stage_objects(env, stage_annotations[2]),
            "pickup_popcorn_from_bar": get_stage_objects(env, stage_annotations[3]),
            "return_to_microwave": get_stage_objects(env, stage_annotations[4]),
            "place_popcorn_in_microwave": get_stage_objects(env, stage_annotations[5]),
            "close_microwave_door": get_stage_objects(env, stage_annotations[6]),
            "turn_on_microwave": get_stage_objects(env, stage_annotations[7]),
        }

        self._microwave_obj = (
            self._stage_objects["move_to_microwave"][0] if self._stage_objects["move_to_microwave"] else None
        )
        self._popcorn_obj = self._stage_objects["move_to_popcorn"][0] if self._stage_objects["move_to_popcorn"] else None
        self._support_obj = (
            self._stage_objects["pickup_popcorn_from_bar"][1]
            if len(self._stage_objects["pickup_popcorn_from_bar"]) > 1
            else None
        )
        self._open_state = self._microwave_obj.states[Open] if self._microwave_obj is not None else None
        self._toggle_state = self._microwave_obj.states[ToggledOn] if self._microwave_obj is not None else None
        self._has_left_support = False
        self._has_picked_up = False
        super().reset(task, env)

    def _build_stages(self, task, env):
        if self._microwave_obj is None or self._popcorn_obj is None or self._open_state is None or self._toggle_state is None:
            return [{"name": "missing_target"}]

        return [
            {
                "name": "move_to_microwave",
                "objects": self._stage_objects.get("move_to_microwave", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "open_microwave_door",
                "objects": self._stage_objects.get("open_microwave_door", []),
                "state": {"prev_distance": None, "was_open": False},
            },
            {
                "name": "move_to_popcorn",
                "objects": self._stage_objects.get("move_to_popcorn", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "pickup_popcorn_from_bar",
                "objects": self._stage_objects.get("pickup_popcorn_from_bar", []),
                "state": {"prev_eef_distance": None},
            },
            {
                "name": "return_to_microwave",
                "objects": self._stage_objects.get("return_to_microwave", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "place_popcorn_in_microwave",
                "objects": self._stage_objects.get("place_popcorn_in_microwave", []),
                "state": {"prev_eef_distance": None, "was_inside": False},
            },
            {
                "name": "close_microwave_door",
                "objects": self._stage_objects.get("close_microwave_door", []),
                "state": {"prev_distance": None, "was_open": True},
            },
            {
                "name": "turn_on_microwave",
                "objects": self._stage_objects.get("turn_on_microwave", []),
                "state": {"prev_distance": None, "prev_toggle_steps": None},
            },
        ]

    def _evaluate_stage(self, stage, task, env, action):
        del task, action
        if self._microwave_obj is None or self._popcorn_obj is None or self._open_state is None or self._toggle_state is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = env.robots[0]
        stage_state = stage["state"]
        stage_name = stage["name"]

        microwave_open = self._is_microwave_open()
        popcorn_in_hand = is_target_in_hand(robot, self._popcorn_obj)
        popcorn_inside = Inside in self._popcorn_obj.states and bool(self._popcorn_obj.states[Inside].get_value(self._microwave_obj))
        microwave_on = bool(self._toggle_state.get_value())

        if stage_name == "move_to_microwave":
            distance = get_min_eef_distance_to_obj(robot, self._microwave_obj)
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
                    "eef_to_microwave_distance": distance,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "open_microwave_door":
            distance = get_min_eef_distance_to_obj(robot, self._microwave_obj)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.open_progress_scale, invert=True
            )
            if microwave_open and not stage_state["was_open"]:
                progress_reward += self.open_progress_scale
            dense_reward = self._exp_distance_reward(distance, self.open_dense_scale)
            stage_state["prev_distance"] = distance
            stage_state["was_open"] = microwave_open
            return {
                "reward": progress_reward + dense_reward,
                "completed": microwave_open,
                "metrics": {
                    "eef_to_microwave_distance": distance,
                    "microwave_open": microwave_open,
                },
            }

        if stage_name == "move_to_popcorn":
            distance = get_min_eef_distance_to_obj(robot, self._popcorn_obj)
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
                    "eef_to_popcorn_distance": distance,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "pickup_popcorn_from_bar":
            distance = get_min_eef_distance_to_obj(robot, self._popcorn_obj)
            on_support = is_supported_by_surface(self._popcorn_obj, self._support_obj)
            self._has_left_support = self._has_left_support or (not on_support)
            self._has_picked_up = self._has_picked_up or (self._has_left_support and popcorn_in_hand)
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.pickup_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = distance
            return {
                "reward": progress_reward + dense_reward,
                "completed": self._has_picked_up,
                "metrics": {
                    "eef_to_popcorn_distance": distance,
                    "in_hand": popcorn_in_hand,
                    "on_support": on_support,
                    "has_left_support": self._has_left_support,
                    "has_picked_up": self._has_picked_up,
                },
            }

        if stage_name == "return_to_microwave":
            distance = get_min_eef_distance_to_obj(robot, self._microwave_obj)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.return_to_microwave_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.return_to_microwave_dense_scale)
            stage_state["prev_distance"] = distance
            completed = distance <= self.move_to_success_threshold
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_microwave_distance": distance,
                    "in_hand": popcorn_in_hand,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if stage_name == "place_popcorn_in_microwave":
            distance = min(
                get_min_eef_distance_to_obj(robot, self._popcorn_obj),
                get_min_eef_distance_to_obj(robot, self._microwave_obj),
            )
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.place_in_progress_scale, invert=True
            )
            if popcorn_inside and not stage_state["was_inside"]:
                progress_reward += self.place_in_progress_scale
            dense_reward = self._exp_distance_reward(distance, self.place_in_dense_scale)
            stage_state["prev_eef_distance"] = distance
            stage_state["was_inside"] = popcorn_inside
            completed = popcorn_inside and (not popcorn_in_hand)
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_place_target_distance": distance,
                    "in_hand": popcorn_in_hand,
                    "popcorn_inside_microwave": popcorn_inside,
                    "microwave_open": microwave_open,
                },
            }

        if stage_name == "close_microwave_door":
            distance = get_min_eef_distance_to_obj(robot, self._microwave_obj)
            microwave_closed = not bool(self._open_state.get_value())
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], distance, self.close_progress_scale, invert=True
            )
            if microwave_closed and stage_state["was_open"]:
                progress_reward += self.close_progress_scale
            dense_reward = self._exp_distance_reward(distance, self.close_dense_scale)
            stage_state["prev_distance"] = distance
            stage_state["was_open"] = not microwave_closed
            completed = microwave_closed and popcorn_inside
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_microwave_distance": distance,
                    "microwave_open": not microwave_closed,
                    "popcorn_inside_microwave": popcorn_inside,
                },
            }

        if stage_name == "turn_on_microwave":
            adjusted_distance = get_min_eef_distance_to_toggle(robot, self._microwave_obj, self._toggle_state)
            toggle_steps = int(self._toggle_state.robot_can_toggle_steps)
            toggle_progress_ratio = min(toggle_steps / max(self._toggle_steps_required, 1), 1.0)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], adjusted_distance, self.press_progress_scale, invert=True
            ) + self._progress_reward(
                stage_state["prev_toggle_steps"], toggle_steps, self.toggle_progress_scale, invert=False
            )
            dense_reward = self._exp_distance_reward(adjusted_distance, self.press_dense_scale) + (
                toggle_progress_ratio * self.toggle_progress_dense_scale
            )
            stage_state["prev_distance"] = adjusted_distance
            stage_state["prev_toggle_steps"] = toggle_steps
            return {
                "reward": progress_reward + dense_reward,
                "completed": microwave_on,
                "metrics": {
                    "eef_to_toggle_distance": adjusted_distance,
                    "toggle_steps": toggle_steps,
                    "microwave_open": microwave_open,
                    "popcorn_inside_microwave": popcorn_inside,
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {}}
