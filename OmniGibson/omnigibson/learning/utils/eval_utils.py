from copy import deepcopy
import os
import json
import csv
from typing import Dict, List
import numpy as np
import torch as th
from collections import OrderedDict


ROBOT_CAMERA_NAMES = {
    "A1": {
        "external": "external::external_camera",
        "wrist": "external::wrist_camera",
    },
    "R1Pro": {
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1::robot_r1:zed_link:Camera:0",
    },
}

# Camera resolutions and corresponding intrinstics
HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)
# TODO: Fix A1
CAMERA_INTRINSICS = {
    "A1": {
        "external": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 240x240
        "wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 240x240
    },
    "R1Pro": {
        "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
    },
}


# Action indices
ACTION_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:7],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    ),
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "A1": OrderedDict(
        {
            "joint_qpos": np.s_[0:8],
            "joint_qpos_sin": np.s_[8:16],
            "joint_qpos_cos": np.s_[16:24],
            "joint_qvel": np.s_[24:32],
            "joint_qeffort": np.s_[32:40],
            "eef_0_pos": np.s_[40:43],
            "eef_0_quat": np.s_[43:47],
            "grasp_0": np.s_[47:48],
            "gripper_0_qpos": np.s_[48:50],
            "gripper_0_qvel": np.s_[50:52],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[
                0:28
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_sin": np.s_[
                28:56
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_cos": np.s_[
                56:84
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],  # Global pos, this is NOT allowed in standard track
            "robot_ori_cos": np.s_[143:146],  # Global ori, this is NOT allowed in standard track
            "robot_ori_sin": np.s_[146:149],  # Global ori, this is NOT allowed in standard track
            "robot_2d_ori": np.s_[149:150],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_cos": np.s_[150:151],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_sin": np.s_[151:152],  # 2D global ori, this is NOT allowed in standard track
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_sin": np.s_[247:250],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_cos": np.s_[250:253],  # Base joint position, this is NOT allowed in standard track
            "base_qvel": np.s_[253:256],
        }
    ),
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:8],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    ),
}


# Joint limits (lower, upper)
JOINT_RANGE = {
    "A1": {
        "arm": (
            th.tensor([-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798], dtype=th.float32),
            th.tensor([2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798], dtype=th.float32),
        ),
        "gripper": (th.tensor([0.00], dtype=th.float32), th.tensor([0.03], dtype=th.float32)),
    },
    "R1Pro": {
        "base": (th.tensor([-0.75, -0.75, -1.0], dtype=th.float32), th.tensor([0.75, 0.75, 1.0], dtype=th.float32)),
        "torso": (
            th.tensor([-1.1345, -2.7925, -1.8326, -3.0543], dtype=th.float32),
            th.tensor([1.8326, 2.5307, 1.5708, 3.0543], dtype=th.float32),
        ),
        "left_arm": (
            th.tensor([-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=th.float32),
            th.tensor([1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=th.float32),
        ),
        "left_gripper": (th.tensor([-1], dtype=th.float32), th.tensor([1], dtype=th.float32)),
        "right_arm": (
            th.tensor([-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=th.float32),
            th.tensor([1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=th.float32),
        ),
        "right_gripper": (th.tensor([0.00], dtype=th.float32), th.tensor([0.05], dtype=th.float32)),
    },
}


EEF_POSITION_RANGE = {
    "A1": {
        "0": (th.tensor([0.0, -0.7, 0.0], dtype=th.float32), th.tensor([0.7, 0.7, 0.7], dtype=th.float32)),
    },
    "R1Pro": {
        "left": (th.tensor([0.0, -0.65, 0.0], dtype=th.float32), th.tensor([0.65, 0.65, 2.5], dtype=th.float32)),
        "right": (th.tensor([0.0, -0.65, 0.0], dtype=th.float32), th.tensor([0.65, 0.65, 2.5], dtype=th.float32)),
    },
}


TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_INDICES_TO_NAMES = {v: k for k, v in TASK_NAMES_TO_INDICES.items()}


def load_subtask_frame(orchestrators_annotation_dir, subtask_index, is_start_frame=True):
    annotation_path = os.path.join(orchestrators_annotation_dir, f"subtask_{int(subtask_index)}_annotated.json")
    with open(annotation_path, "r") as f:
        subtask_info = json.load(f)
    frame = subtask_info.get("start_frame") if is_start_frame else subtask_info.get("end_frame")
    
    assert isinstance(frame, int), (
        f"Subtask annotation {annotation_path} is missing an integer frame: {frame}"
    )
    return frame


def extract_sequential_reward_info(info: Dict) -> Dict:
    if not isinstance(info, dict):
        return {}

    if "stage_infos" in info or "current_stage_name" in info:
        return info

    reward_info = info.get("reward")
    if isinstance(reward_info, dict):
        task_specific = reward_info.get("task_specific")
        if isinstance(task_specific, dict):
            return task_specific

        for reward_payload in reward_info.values():
            if isinstance(reward_payload, dict) and (
                "stage_infos" in reward_payload or "current_stage_name" in reward_payload
            ):
                return reward_payload

    return {}


def delay_termination_until_stage_completion(info: Dict) -> Dict:
    info = deepcopy(info) if isinstance(info, dict) else {}
    sequential_info = extract_sequential_reward_info(info)
    done_info = info.get("done")
    if not isinstance(done_info, dict) or not sequential_info:
        return info

    if done_info.get("success"):
        if not sequential_info.get("all_stages_completed", False):
            done_info["success"] = False
            done_info["waiting_for_stage_completion"] = True
        else:
            done_info["keep_running_after_success"] = True

    return info


def get_task_specific_reward(evaluator):
    reward_functions = getattr(evaluator.env.task, "_reward_functions", {})
    task_reward = reward_functions.get("task_specific") if isinstance(reward_functions, dict) else None
    assert task_reward is not None, (
        "Subtask reward-stage evaluation requires a task_specific reward. "
        "Set instance_reward_mode=task or combined and provide a task-specific reward implementation."
    )
    return task_reward


def get_reward_stage_result(info: Dict, target_stage_idx: int):
    reward_info = info['reward']
    stage_infos = reward_info['task_specific']['stage_infos']
    stage_name = list(stage_infos.keys())[target_stage_idx]
    stage_info = stage_infos[stage_name]
    stage_completed = bool(stage_info['completed'])
    return stage_completed, stage_info, reward_info


def prime_task_reward_for_subtask(evaluator, subtask_idx: int) -> None:
    task_reward = get_task_specific_reward(evaluator)
    if hasattr(task_reward, "set_active_stage_index"):
        # When we jump into subtask i from demo state, earlier reward stages
        # should already count as finished so logs and completion checks align.
        task_reward.set_active_stage_index(subtask_idx)


def format_subtask_range_label(start_idx: int, end_idx: int) -> str:
    return f"{start_idx}" if start_idx == end_idx else f"{start_idx}->{end_idx}"


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


_VIDEO_ACTIVE_METRIC_KEYS = {
    "move_to_radio": ["eef_to_obj_distance", "success_threshold"],
    "pickup_from_support": ["eef_to_obj_distance"],
    "press_radio": ["eef_to_toggle_distance", "toggle_steps"],
    "place_on_support": ["eef_to_obj_distance"],
}


def format_stage_status_chain(info: Dict):
    info = extract_sequential_reward_info(info)
    stage_infos = info.get("stage_infos")
    current_stage_name = info.get("current_stage_name")
    all_stages_completed = bool(info.get("all_stages_completed", False))
    if not isinstance(stage_infos, dict) or len(stage_infos) == 0:
        return None

    parts = []
    for stage_name, stage_info in stage_infos.items():
        if all_stages_completed or bool(stage_info.get("completed", False)):
            status = "done"
        elif stage_name == current_stage_name:
            status = "doing"
        else:
            status = "todo"
        parts.append(f"{stage_name} ({status})")

    return " > ".join(parts)


def format_stage_progress_lines(info: Dict, *, concise: bool=False) -> List[str]:
    reward_info = extract_sequential_reward_info(info or {})
    lines = []
    stage_chain = format_stage_status_chain(reward_info)
    if stage_chain is not None:
        lines.append(f"stages: {stage_chain}")

    stage_total_rewards = reward_info.get("stage_cumulative_rewards")
    stage_rewards = reward_info.get("stage_rewards")
    if isinstance(stage_total_rewards, dict) and len(stage_total_rewards) > 0:
        lines.append(
            "stage_total_rewards: "
            + ", ".join(
                f"{stage_name}={stage_reward:.3f}" for stage_name, stage_reward in stage_total_rewards.items()
            )
        )
    elif isinstance(stage_rewards, dict) and len(stage_rewards) > 0:
        lines.append(
            "stage_rewards: "
            + ", ".join(f"{stage_name}={stage_reward:.3f}" for stage_name, stage_reward in stage_rewards.items())
        )

    current_stage_name = reward_info.get("current_stage_name")
    stage_infos = reward_info.get("stage_infos")
    active_stage_info = stage_infos.get(current_stage_name, {}) if isinstance(stage_infos, dict) else {}
    if isinstance(current_stage_name, str) and isinstance(active_stage_info, dict):
        stage_reward = active_stage_info.get("reward")
        if not isinstance(stage_reward, (int, float)) or isinstance(stage_reward, bool):
            stage_reward = stage_rewards.get(current_stage_name) if isinstance(stage_rewards, dict) else None
        if isinstance(stage_reward, (int, float)) and not isinstance(stage_reward, bool):
            lines.append(f"reward: {_format_scalar(stage_reward)}")

        completed = active_stage_info.get("completed")
        if isinstance(completed, bool):
            lines.append(f"completed: {completed}")

        metric_keys = set(_VIDEO_ACTIVE_METRIC_KEYS.get(current_stage_name, [])) if concise else None
        metrics = [
            f"{key}={_format_scalar(value)}"
            for key, value in active_stage_info.items()
            if key not in {"reward", "completed"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (metric_keys is None or key in metric_keys)
        ]
        if metrics:
            lines.append("metrics: " + ", ".join(metrics))

    return lines


def format_video_info_lines(info: Dict, step: int, reward: float) -> List[str]:
    return [f"step={step} reward={reward:.4f}", *format_stage_progress_lines(info, concise=True)]


def generate_basic_environment_config(task_name, task_cfg):
    """
    Generate a basic environment configuration

    Args:
        task_name (str): Name of the task
        task_cfg: Dictionary of task config

    Returns:
        dict: Environment configuration
    """
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": task_cfg["scene_model"],
            "load_room_types": None,
            "load_room_instances": task_cfg.get("load_room_instances", None),
            "include_robots": False,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {
                "max_steps": 5000,
            },
            "reward_config": {
                "r_potential": 1.0,
            },
            "include_obs": False,
        },
    }
    return cfg


def flatten_obs_dict(obs: dict, parent_key: str = "") -> dict:
    """
    Process the observation dictionary by recursively flattening the keys.
    so obs["robot_r1"]["camera"]["rgb"] will become obs["robot_r1::camera:::rgb"].
    """
    processed_obs = {}
    for key, value in obs.items():
        new_key = f"{parent_key}::{key}" if parent_key else key
        if isinstance(value, dict):
            processed_obs.update(flatten_obs_dict(value, parent_key=new_key))
        else:
            processed_obs[new_key] = value
    return processed_obs


def find_start_point(base_vel):
    """
    Find the first point where the base velocity is non-zero.
    This is used to skip the initial part of the dataset where the robot is not moving.
    """
    start_idx = np.where(np.linalg.norm(base_vel, axis=-1) > 1e-5)[0]
    if len(start_idx) == 0:
        return 0
    return min(start_idx[0], 500)  # Limit to the first 100 points to avoid long initial periods


def get_instance_to_run(config, m, gm, logger):
    # get run instances
    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
    elif config.test_hidden:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
    else:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]
    return instances_to_run