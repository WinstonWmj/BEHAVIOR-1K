from omnigibson.object_states.toggle import ToggledOn, m as toggle_macros
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    load_orchestrator_stage_annotations,
    get_stage_objects,
    get_min_eef_distance_to_obj,
    get_min_eef_distance_to_toggle,
    is_supported_by_surface,
    is_target_in_hand,
)


class TurningOnRadioReward(SequentialTaskReward):
    """Task-bound sequential reward for `turning_on_radio`."""

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=5.0,
        move_to_dense_scale=0.3,
        pickup_progress_scale=3.0,
        pickup_dense_scale=0.35,
        press_progress_scale=6.0,
        press_dense_scale=0.25,
        toggle_progress_scale=0.5,
        toggle_progress_dense_scale=0.4,
        placedown_progress_scale=3.0,
        placedown_dense_scale=0.25,
        stage_completion_bonus=1.0,
        orchestrators_annotation_dir=None,
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.press_progress_scale = press_progress_scale
        self.press_dense_scale = press_dense_scale
        self.toggle_progress_scale = toggle_progress_scale
        self.toggle_progress_dense_scale = toggle_progress_dense_scale
        self.placedown_progress_scale = placedown_progress_scale
        self.placedown_dense_scale = placedown_dense_scale
        self.orchestrators_annotation_dir = orchestrators_annotation_dir
        self._radio_obj = None
        self._toggle_state = None
        self._support_obj = None
        self._stage_objects = {}
        self._has_left_support = False
        self._has_picked_up = False
        self._toggle_steps_required = int(getattr(toggle_macros, "CAN_TOGGLE_STEPS", 5))
        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def reset(self, task, env):
        stage_annotations = load_orchestrator_stage_annotations(self.orchestrators_annotation_dir)
        self._stage_objects = {
            "move_to_radio": get_stage_objects(env, stage_annotations[0]),
            "pickup_from_support": get_stage_objects(env, stage_annotations[1]),
            "press_radio": get_stage_objects(env, stage_annotations[2]),
            "place_on_support": get_stage_objects(env, stage_annotations[3]),
        }
        self._radio_obj = self._stage_objects["move_to_radio"][0] if self._stage_objects["move_to_radio"] else None
        self._toggle_state = self._radio_obj.states[ToggledOn] if self._radio_obj is not None else None
        self._support_obj = self._stage_objects["pickup_from_support"][1] if len(self._stage_objects["pickup_from_support"]) > 1 else None
        self._has_left_support = False
        self._has_picked_up = False
        super().reset(task, env)

    def _build_stages(self, task, env):
        if self._radio_obj is None or self._toggle_state is None:
            return [{"name": "missing_target"}]
        return [
            {
                "name": "move_to_radio",
                "objects": self._stage_objects.get("move_to_radio", []),
                "state": {"prev_distance": None},
            },
            {
                "name": "pickup_from_support",
                "objects": self._stage_objects.get("pickup_from_support", []),
                "state": {"prev_eef_distance": None},
            },
            {
                "name": "press_radio",
                "objects": self._stage_objects.get("press_radio", []),
                "state": {"prev_distance": None, "prev_toggle_steps": None},
            },
            {
                "name": "place_on_support",
                "objects": self._stage_objects.get("place_on_support", []),
                "state": {"prev_eef_distance": None},
            },
        ]

    def _evaluate_stage(self, stage, task, env, action):
        if self._radio_obj is None or self._toggle_state is None:
            return {"reward": 0.0, "completed": False, "metrics": {"missing_target": True}}

        robot = env.robots[0]
        stage_state = stage["state"]
        stage_name = stage["name"]
        toggled_on = bool(self._toggle_state.get_value())

        if stage_name == "move_to_radio":
            distance = get_min_eef_distance_to_obj(robot, self._radio_obj)
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
            distance = get_min_eef_distance_to_obj(robot, self._radio_obj)
            in_hand = is_target_in_hand(robot, self._radio_obj)
            on_support = is_supported_by_surface(
                self._radio_obj,
                self._support_obj,
            )
            self._has_left_support = self._has_left_support or (not on_support)
            self._has_picked_up = self._has_picked_up or (self._has_left_support and in_hand)
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.pickup_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.pickup_dense_scale)
            stage_state["prev_eef_distance"] = distance
            completed = self._has_picked_up
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                    "has_left_support": self._has_left_support,
                    "has_picked_up": self._has_picked_up,
                },
            }

        if stage_name == "press_radio":
            adjusted_distance = get_min_eef_distance_to_toggle(robot, self._radio_obj, self._toggle_state)
            # `robot_can_toggle_steps` comes from the ToggledOn state and counts how many consecutive
            # simulator updates the robot's fingers are in valid toggle contact with the button area.
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
            completed = toggled_on
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_toggle_distance": adjusted_distance,
                    "toggle_steps": toggle_steps,
                },
            }

        if stage_name == "place_on_support":
            distance = get_min_eef_distance_to_obj(robot, self._radio_obj)
            in_hand = is_target_in_hand(robot, self._radio_obj)
            on_support = is_supported_by_surface(
                self._radio_obj,
                self._support_obj,
            )
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.placedown_progress_scale, invert=True
            )
            dense_reward = self._exp_distance_reward(distance, self.placedown_dense_scale)
            stage_state["prev_eef_distance"] = distance
            completed = on_support and (not in_hand)
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                },
            }

        return {"reward": 0.0, "completed": False, "metrics": {}}
