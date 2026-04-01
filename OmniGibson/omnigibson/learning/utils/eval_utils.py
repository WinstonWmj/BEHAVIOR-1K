from copy import deepcopy
import json
import logging
import os
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


def get_demo_annotation_path(demo_data_dir, task_index, episode_index):
    return os.path.join(
        demo_data_dir,
        "annotations",
        f"task-{task_index:04d}",
        f"episode_{int(episode_index):08d}.json",
    )


def resolve_demo_annotation_path(demo_data_dir, task_index, episode_index):
    annotation_path = (
        get_demo_annotation_path(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
        )
        if demo_data_dir is not None
        else None
    )
    return annotation_path if annotation_path is not None and os.path.exists(annotation_path) else None


def get_subtask_annotation_path(demo_data_dir, task_index, episode_index, subtask_index):
    return os.path.join(
        demo_data_dir,
        "orchestrators",
        f"task-{task_index:04d}",
        f"episode_{int(episode_index):08d}",
        f"subtask_{int(subtask_index)}_annotated.json",
    )


def load_subtask_annotation(demo_data_dir, task_index, episode_index, subtask_index):
    annotation_path = get_subtask_annotation_path(
        demo_data_dir=demo_data_dir,
        task_index=task_index,
        episode_index=episode_index,
        subtask_index=subtask_index,
    )
    assert os.path.exists(annotation_path), f"Subtask annotation not found: {annotation_path}"
    with open(annotation_path, "r") as f:
        return annotation_path, json.load(f)


def resolve_subtask_index_range(subtask_index=None, subtask_end_index=None):
    if subtask_index is None:
        assert subtask_end_index is None, "subtask_end_index requires subtask_index to also be set."
        return None

    start_idx = int(subtask_index)
    end_idx = start_idx if subtask_end_index is None else int(subtask_end_index)
    assert end_idx >= start_idx, (
        f"Expected subtask_end_index >= subtask_index, got start={start_idx}, end={end_idx}"
    )
    return start_idx, end_idx


def resolve_subtask_frame_range(demo_data_dir, task_index, episode_index, subtask_index, subtask_end_index=None):
    """
    Resolve the inclusive-exclusive frame range [start_frame, end_frame) for a
    contiguous subtask selection. When only subtask_index is provided, this
    resolves a single annotated subtask.
    """
    resolved_range = resolve_subtask_index_range(
        subtask_index=subtask_index,
        subtask_end_index=subtask_end_index,
    )
    assert resolved_range is not None, "subtask_index must be set to resolve a subtask frame range."
    start_subtask_index, end_subtask_index = resolved_range

    annotation_path, subtask_info = load_subtask_annotation(
        demo_data_dir=demo_data_dir,
        task_index=task_index,
        episode_index=episode_index,
        subtask_index=start_subtask_index,
    )
    end_annotation_path, end_subtask_info = load_subtask_annotation(
        demo_data_dir=demo_data_dir,
        task_index=task_index,
        episode_index=episode_index,
        subtask_index=end_subtask_index,
    )
    start_frame = subtask_info.get("start_frame")
    end_frame = end_subtask_info.get("end_frame")
    assert isinstance(start_frame, int), (
        f"Subtask annotation {annotation_path} is missing an integer start_frame: {start_frame}"
    )
    assert isinstance(end_frame, int), (
        f"Subtask annotation {end_annotation_path} is missing an integer end_frame: {end_frame}"
    )
    assert start_frame < end_frame, (
        f"Subtask selection task={task_index}, episode={episode_index}, subtasks=[{start_subtask_index}, {end_subtask_index}] "
        f"has invalid frame range: "
        f"start_frame={start_frame}, end_frame={end_frame}"
    )
    return start_frame, end_frame


def sync_task_reward_annotation_for_episode(task, demo_data_dir, task_index, episode_index, logger=None):
    orchestrator_annotation_path = resolve_demo_annotation_path(
        demo_data_dir=demo_data_dir,
        task_index=task_index,
        episode_index=episode_index,
    )
    reward_function = getattr(task, "_reward_functions", {}).get("task_specific", None)
    if reward_function is None or not hasattr(reward_function, "orchestrator_annotation_path"):
        return None

    task_reward_kwargs = task._reward_config.get("task_specific_reward_kwargs", {})
    reward_function.orchestrator_annotation_path = orchestrator_annotation_path
    if orchestrator_annotation_path is None:
        task_reward_kwargs.pop("orchestrator_annotation_path", None)
    else:
        task_reward_kwargs["orchestrator_annotation_path"] = orchestrator_annotation_path

    active_logger = logger or logging.getLogger(__name__)
    if orchestrator_annotation_path is not None:
        active_logger.info(
            "Using task reward orchestrator annotation for current episode: %s",
            orchestrator_annotation_path,
        )
    else:
        missing_path = get_demo_annotation_path(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
        )
        active_logger.warning("Task reward orchestrator annotation not found for current episode: %s", missing_path)
    return orchestrator_annotation_path


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


def _format_stage_progress_lines(info: Dict, *, concise: bool) -> List[str]:
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
    return [f"step={step} reward={reward:.4f}", *_format_stage_progress_lines(info, concise=True)]


def summarize_stage_progress(info: Dict) -> List[str]:
    return _format_stage_progress_lines(info, concise=False)


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
