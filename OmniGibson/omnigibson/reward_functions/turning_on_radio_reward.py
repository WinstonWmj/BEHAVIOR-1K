import math

import torch as th

from omnigibson.object_states.toggle import ToggledOn, m as toggle_macros
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    get_stage_objects_by_name,
    get_min_eef_distance_to_obj,
    get_min_eef_distance_to_toggle,
    is_supported_by_surface,
    is_target_in_hand,
)


class TurningOnRadioReward(SequentialTaskReward):
    """Task-bound sequential reward for `turning_on_radio`."""

    STAGE_OBJECT_NAMES = {
        "move_to_radio": ("radio_89",),
        "pickup_from_support": ("radio_89", "coffee_table_koagbh_0"),
        "press_radio": ("radio_89",),
        "place_on_support": ("radio_89", "coffee_table_koagbh_0"),
    }

    def __init__(
        self,
        move_to_success_threshold=0.3,
        move_to_progress_scale=0.05,
        move_to_dense_scale=0.003,
        pickup_progress_scale=0.03,
        pickup_dense_scale=0.0035,
        pickup_lift_success_threshold=0.30,
        pickup_orientation_reward_start_threshold=0.10,
        pickup_button_up_angle_threshold=math.radians(60.0),
        pickup_lift_dense_scale=0.0035,
        pickup_orientation_dense_scale=0.0035,
        press_progress_scale=0.06,
        press_dense_scale=0.0025,
        toggle_progress_scale=0.005,
        toggle_progress_dense_scale=0.004,
        placedown_progress_scale=0.03,
        placedown_dense_scale=0.0025,
        stage_completion_bonus=1.0,
        reward_mode="task",
    ):
        self.move_to_success_threshold = move_to_success_threshold
        self.move_to_progress_scale = move_to_progress_scale
        self.move_to_dense_scale = move_to_dense_scale
        self.pickup_progress_scale = pickup_progress_scale
        self.pickup_dense_scale = pickup_dense_scale
        self.pickup_lift_success_threshold = pickup_lift_success_threshold
        self.pickup_orientation_reward_start_threshold = pickup_orientation_reward_start_threshold
        self.pickup_button_up_alignment_threshold = math.cos(pickup_button_up_angle_threshold)
        self.pickup_lift_dense_scale = pickup_lift_dense_scale
        self.pickup_orientation_dense_scale = pickup_orientation_dense_scale
        self.press_progress_scale = press_progress_scale
        self.press_dense_scale = press_dense_scale
        self.toggle_progress_scale = toggle_progress_scale
        self.toggle_progress_dense_scale = toggle_progress_dense_scale
        self.placedown_progress_scale = placedown_progress_scale
        self.placedown_dense_scale = placedown_dense_scale
        self._radio_obj = None
        self._toggle_state = None
        self._support_obj = None
        self._stage_objects = {}
        self._has_left_support = False
        self._has_picked_up = False
        self._radio_initial_pos = None
        self._toggle_steps_required = int(getattr(toggle_macros, "CAN_TOGGLE_STEPS", 5))
        super().__init__(stage_completion_bonus=stage_completion_bonus, reward_mode=reward_mode)

    def reset(self, task, env):
        self._stage_objects = {
            stage_name: get_stage_objects_by_name(env, object_names)
            for stage_name, object_names in self.STAGE_OBJECT_NAMES.items()
        }
        self._radio_obj = self._stage_objects["move_to_radio"][0] if self._stage_objects["move_to_radio"] else None
        self._toggle_state = self._radio_obj.states[ToggledOn] if self._radio_obj is not None else None
        self._support_obj = self._stage_objects["pickup_from_support"][1] if len(self._stage_objects["pickup_from_support"]) > 1 else None
        self._has_left_support = False
        self._has_picked_up = False
        self._radio_initial_pos = self._radio_obj.get_position_orientation()[0].clone() if self._radio_obj is not None else None
        super().reset(task, env)

    def _get_radio_displacement_from_initial(self):
        if self._radio_obj is None or self._radio_initial_pos is None:
            return 0.0
        radio_pos = self._radio_obj.get_position_orientation()[0]
        return th.norm(radio_pos - self._radio_initial_pos).item()

    def _get_radio_front_up_alignment(self):
        if self._radio_obj is None or self._toggle_state is None:
            return 0.0

        try:
            button_pos = self._toggle_state.link.get_position_orientation()[0]
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            visual_marker = getattr(self._toggle_state, "visual_marker", None)
            if visual_marker is None:
                return 0.0
            button_pos = visual_marker.get_position_orientation()[0]

        radio_pos = getattr(self._radio_obj, "aabb_center", None)
        if radio_pos is None:
            radio_pos = self._radio_obj.get_position_orientation()[0]

        front_direction = button_pos - radio_pos
        front_direction = front_direction / th.clamp(th.norm(front_direction), min=1e-6)
        world_up = th.tensor([0.0, 0.0, 1.0], dtype=front_direction.dtype, device=front_direction.device)
        return th.clamp(th.dot(front_direction, world_up), min=-1.0, max=1.0).item()

    def _get_radio_front_up_metrics(self, orientation_reward_active):
        radio_front_up_alignment = self._get_radio_front_up_alignment()
        raw_radio_front_up = radio_front_up_alignment >= self.pickup_button_up_alignment_threshold
        radio_front_up = orientation_reward_active and raw_radio_front_up

        orientation_ratio = 0.0
        if orientation_reward_active:
            if radio_front_up_alignment >= self.pickup_button_up_alignment_threshold:
                orientation_ratio = (
                    (radio_front_up_alignment - self.pickup_button_up_alignment_threshold)
                    / max(1.0 - self.pickup_button_up_alignment_threshold, 1e-6)
                )
            elif radio_front_up_alignment <= -self.pickup_button_up_alignment_threshold:
                orientation_ratio = -(
                    (-self.pickup_button_up_alignment_threshold - radio_front_up_alignment)
                    / max(1.0 + self.pickup_button_up_alignment_threshold, 1e-6)
                )

        return {
            "button_up_alignment": radio_front_up_alignment,
            "button_up_alignment_threshold": self.pickup_button_up_alignment_threshold,
            "button_normal_up": radio_front_up,
            "raw_button_normal_up": raw_radio_front_up,
            "radio_front_up_alignment": radio_front_up_alignment,
            "raw_radio_front_up": raw_radio_front_up,
            "orientation_reward_ratio": orientation_ratio,
            "orientation_reward_active": orientation_reward_active,
        }

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
            radio_displacement = self._get_radio_displacement_from_initial()
            lifted_enough = radio_displacement >= self.pickup_lift_success_threshold
            orientation_reward_active = (
                self._has_picked_up
                and radio_displacement >= self.pickup_orientation_reward_start_threshold
            )
            orientation_metrics = self._get_radio_front_up_metrics(orientation_reward_active)
            progress_reward = self._progress_reward(
                stage_state["prev_eef_distance"], distance, self.pickup_progress_scale, invert=True
            )
            lift_ratio = min(radio_displacement / max(self.pickup_lift_success_threshold, 1e-6), 1.0)
            dense_reward = (
                self._exp_distance_reward(distance, self.pickup_dense_scale)
                + lift_ratio * self.pickup_lift_dense_scale
                + orientation_metrics["orientation_reward_ratio"] * self.pickup_orientation_dense_scale
            )
            stage_state["prev_eef_distance"] = distance
            completed = self._has_picked_up and lifted_enough
            return {
                "reward": progress_reward + dense_reward,
                "completed": completed,
                "metrics": {
                    "eef_to_obj_distance": distance,
                    "in_hand": in_hand,
                    "on_support": on_support,
                    "has_left_support": self._has_left_support,
                    "has_picked_up": self._has_picked_up,
                    "radio_displacement_from_initial": radio_displacement,
                    "pickup_lift_success_threshold": self.pickup_lift_success_threshold,
                    "lifted_enough": lifted_enough,
                    **orientation_metrics,
                },
            }

        if stage_name == "press_radio":
            adjusted_distance = get_min_eef_distance_to_toggle(robot, self._radio_obj, self._toggle_state)
            # `robot_can_toggle_steps` comes from the ToggledOn state and counts how many consecutive
            # simulator updates the robot's fingers are in valid toggle contact with the button area.
            toggle_steps = int(self._toggle_state.robot_can_toggle_steps)
            toggle_progress_ratio = min(toggle_steps / max(self._toggle_steps_required, 1), 1.0)
            orientation_metrics = self._get_radio_front_up_metrics(orientation_reward_active=self._has_picked_up)
            progress_reward = self._progress_reward(
                stage_state["prev_distance"], adjusted_distance, self.press_progress_scale, invert=True
            ) + self._progress_reward(
                stage_state["prev_toggle_steps"], toggle_steps, self.toggle_progress_scale, invert=False
            )
            dense_reward = self._exp_distance_reward(adjusted_distance, self.press_dense_scale) + (
                toggle_progress_ratio * self.toggle_progress_dense_scale
            ) + (
                orientation_metrics["orientation_reward_ratio"] * self.pickup_orientation_dense_scale
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
                    **orientation_metrics,
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
