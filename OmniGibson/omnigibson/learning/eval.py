import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import pandas as pd
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    delay_termination_until_stage_completion,
    extract_sequential_reward_info,
    generate_basic_environment_config,
    flatten_obs_dict,
    get_demo_annotation_path,
    resolve_demo_annotation_path,
    resolve_subtask_frame_range,
    resolve_subtask_index_range,
    summarize_stage_progress,
    sync_task_reward_annotation_for_episode,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    overlay_info_banner,
    write_video,
)
from omnigibson.macros import gm, create_module_macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple, List

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Set grasp window to larger value to account for hard grasps
with gm.unlocked():
    gm.robots.manipulation_robot.GRASP_WINDOW = 0.75


# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


class Evaluator:
    """
    Evaluator class for running and evaluating policies for behavior task.
    This class manages the setup, execution, and evaluation of policy rollouts in OmniGibson environment,
    tracking metrics such as the number of trials, successes, and total time. It supports loading environments,
    robots, policies, and metrics, and provides methods for stepping through the environment, resetting state,
    and handling video outputs and loggings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None
        self.last_step_info = {}
        self.last_step_reward = 0.0
        self.last_policy_done = False

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        cfg["task"]["reward_config"]["reward_mode"] = self.cfg.instance_reward_mode
        if self.cfg.instance_reward_mode in {"task", "combined"}:
            task_reward_kwargs = OmegaConf.to_container(self.cfg.task_specific_reward_kwargs, resolve=True) or {}
            demo_expert_data_dir = self.cfg.demo_expert_data_dir
            demo_expert_episode_index = self.cfg.demo_expert_episode_index
            if demo_expert_data_dir is not None and demo_expert_episode_index is not None:
                annotation_path = resolve_demo_annotation_path(
                    demo_data_dir=demo_expert_data_dir,
                    task_index=task_idx,
                    episode_index=demo_expert_episode_index,
                )
                if annotation_path is not None:
                    task_reward_kwargs["annotation_path"] = annotation_path
                    logger.info("Using task reward annotation: %s", annotation_path)
                else:
                    annotation_path = get_demo_annotation_path(
                        demo_data_dir=demo_expert_data_dir,
                        task_index=task_idx,
                        episode_index=demo_expert_episode_index,
                    )
                    logger.warning("Task reward annotation not found: %s", annotation_path)

            cfg["task"]["reward_config"]["task_specific_reward_name"] = task_name
            cfg["task"]["reward_config"]["task_specific_reward_kwargs"] = task_reward_kwargs
        logger.info(
            "Using reward mode '%s' for task '%s'",
            cfg["task"]["reward_config"]["reward_mode"],
            task_name,
        )
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> Robot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            Robot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """
        Loads and returns the policy instance.
        """
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.

        Returns:
            Tuple[bool, bool]:
                - terminated (bool): Whether the episode has terminated (i.e., reached a terminal state).
                - truncated (bool): Whether the episode was truncated (i.e., stopped due to a time limit or other constraint).

        Workflow:
            1. Computes the next action using the policy based on the current observation.
            2. Steps the environment with the computed action and retrieves the next observation,
               termination and truncation flags, and additional info.
            3. If the episode has ended (terminated or truncated), increments the trial counter and
               updates the count of successful trials if the task was completed successfully.
            4. Preprocesses the new observation.
            5. Invokes step callbacks for all registered metrics to update their state.
            6. Returns the termination and truncation status.
        """
        self.robot_action = self.policy.forward(obs=self.obs)

        obs, reward, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        if terminated and not truncated and (
            self.cfg.waiting_for_stage_completion or self.cfg.keep_running_after_success
        ):
            delayed_info = delay_termination_until_stage_completion(info)
            delayed_done = delayed_info.get("done", {})
            if (
                self.cfg.waiting_for_stage_completion
                and delayed_done.get("waiting_for_stage_completion", False)
            ) or (
                self.cfg.keep_running_after_success
                and delayed_done.get("keep_running_after_success", False)
            ):
                info = delayed_info
                terminated = False
        self.last_step_reward = reward  # reward 已经是基于谓词进度的 delta 奖励（potential-based shaping）新满足一个谓词 → reward > 0，谓词退化 → reward < 0
        self.last_step_info = info
        self.last_policy_done = bool(getattr(self.policy, "is_done", False))
        # process obs
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated, reward, info

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                # tro_state uses "R1Pro"; robot.model is "r1pro"
                robot_pos = presampled_robot_poses["R1Pro"][0]["position"]
                robot_quat = presampled_robot_poses["R1Pro"][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def load_subtask_init_state(self, demo_data_dir: str, task_index: int, episode_index: int, start_frame: int) -> None:
        """
        Load the robot's proprioceptive state from a parquet demo file at a given frame,
        and teleport the robot to the corresponding pose and joint configuration.

        Args:
            demo_data_dir: root path of the 2025-challenge-demos directory.
            task_index: task index (e.g. 13).
            episode_index: full episode index (e.g. 130010).
            start_frame: the frame index in the parquet to restore from.
        """
        parquet_path = os.path.join(
            demo_data_dir, "data", f"task-{task_index:04d}", f"episode_{episode_index:08d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        assert start_frame < len(df), (
            f"start_frame {start_frame} >= parquet length {len(df)} for episode {episode_index}"
        )
        state = df.iloc[start_frame]["observation.state"]

        robot_pos = th.tensor(state[140:143], dtype=th.float32)
        ori_cos = state[143:146]
        ori_sin = state[146:149]
        euler = th.tensor(np.arctan2(ori_sin, ori_cos), dtype=th.float32)
        robot_quat = T.euler2quat(euler)

        joint_qpos_all = th.tensor(state[0:28], dtype=th.float32)

        self.robot.set_position_orientation(robot_pos, robot_quat)
        self.robot.set_joint_positions(joint_qpos_all)

        for _ in range(25):
            og.sim.step_physics()
            self.robot.keep_still()

        og.sim.render()
        self.obs = self._preprocess_obs(self.env.get_obs()[0])

        logger.info(
            f"Loaded subtask init state: episode={episode_index}, frame={start_frame}, "
            f"robot_pos={robot_pos.tolist()}, yaw={euler[2].item():.3f}"
        )

    def find_scene_object(self, obj_name: str):
        """
        Look up a scene object by its annotation name.
        Falls back to searching the task's object_scope if the direct registry
        lookup fails (e.g. for auto-generated instance names).
        """
        try:
            return self.env.scene.object_registry("name", obj_name)
        except Exception:
            pass
        task = self.env.task
        if hasattr(task, "object_scope"):
            for scope_name, scope_obj in task.object_scope.items():
                if obj_name in scope_name or scope_name in obj_name:
                    return scope_obj
        logger.warning(f"Could not find scene object '{obj_name}'")
        return None

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # append task id to obs
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])
        frame = overlay_info_banner(
            frame,
            info=self.last_step_info,
            step=self.env._current_step,
            reward=self.last_step_reward,
        )
        write_video(
            np.expand_dims(frame, 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """
        Reset the environment, policy, and compute metrics.
        """
        self.obs = self._preprocess_obs(self.env.reset()[0])
        self.last_step_info = {}
        self.last_step_reward = 0.0
        self.last_policy_done = False
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


def _get_task_specific_reward(evaluator: Evaluator):
    reward_functions = getattr(evaluator.env.task, "_reward_functions", {})
    task_reward = reward_functions.get("task_specific") if isinstance(reward_functions, dict) else None
    assert task_reward is not None, (
        "Subtask reward-stage evaluation requires a task_specific reward. "
        "Set instance_reward_mode=task or combined and provide a task-specific reward implementation."
    )
    return task_reward


def _resolve_subtask_reward_stage_name(evaluator: Evaluator, subtask_idx: int) -> str:
    task_reward = _get_task_specific_reward(evaluator)
    stage_defs = getattr(task_reward, "_stage_defs", None)
    assert isinstance(stage_defs, list) and len(stage_defs) > 0, (
        "Subtask reward-stage evaluation requires an ordered sequential task reward with non-empty stage_defs."
    )
    assert subtask_idx < len(stage_defs), (
        f"Subtask {subtask_idx} has no matching reward stage. "
        f"Task reward only defines {len(stage_defs)} stages."
    )
    stage_name = stage_defs[subtask_idx].get("name")
    assert stage_name is not None, f"Reward stage definition at index {subtask_idx} is missing a name."
    return stage_name


def _get_reward_stage_result(info: dict, target_stage_name: str) -> Tuple[bool, dict, dict]:
    reward_info = extract_sequential_reward_info(info or {})
    stage_infos = reward_info.get("stage_infos", {}) if isinstance(reward_info, dict) else {}
    stage_info = stage_infos.get(target_stage_name, {}) if isinstance(stage_infos, dict) else {}
    if not isinstance(stage_info, dict):
        stage_info = {}
    stage_completed = bool(stage_info.get("completed", False))
    return stage_completed, stage_info, reward_info


def _prime_task_reward_for_subtask(evaluator: Evaluator, subtask_idx: int) -> None:
    task_reward = _get_task_specific_reward(evaluator)
    if hasattr(task_reward, "set_active_stage_index"):
        # When we jump into subtask i from demo state, earlier reward stages
        # should already count as finished so logs and completion checks align.
        task_reward.set_active_stage_index(subtask_idx)


def _format_subtask_range_label(start_idx: int, end_idx: int) -> str:
    return f"{start_idx}" if start_idx == end_idx else f"{start_idx}->{end_idx}"


def _run_subtask_eval(config, logger):
    """Subtask-level evaluation: iterate episodes × subtasks from demo data."""
    task_idx = TASK_NAMES_TO_INDICES[config.task.name]
    demo_data_dir = config.demo_data_dir
    assert demo_data_dir is not None, "demo_data_dir must be set when eval_level=subtask."
    orchestrator_dir = Path(demo_data_dir) / "orchestrators" / f"task-{task_idx:04d}"
    assert orchestrator_dir.exists(), f"Orchestrator dir not found: {orchestrator_dir}"

    all_episode_dirs = sorted(
        d for d in orchestrator_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    all_episode_indices = [int(d.name.split("_")[1]) for d in all_episode_dirs]

    if config.subtask_episode_indices is not None:
        requested_episode_indices = [int(ep_idx) for ep_idx in config.subtask_episode_indices]
        missing_episodes = sorted(set(requested_episode_indices) - set(all_episode_indices))
        assert not missing_episodes, (
            f"Requested subtask_episode_indices not found under {orchestrator_dir}: {missing_episodes}"
        )
        episodes_to_run = requested_episode_indices
    elif config.eval_instance_ids is not None:
        logger.warning(
            "Using eval_instance_ids to select subtask episodes by position is deprecated. "
            "Please use subtask_episode_indices with full episode ids instead."
        )
        episodes_to_run = [all_episode_indices[i] for i in config.eval_instance_ids]
    else:
        episodes_to_run = all_episode_indices

    skill_filter = None
    if config.subtask_skill_filter is not None:
        skill_filter = set(config.subtask_skill_filter)
    selected_subtask_range = resolve_subtask_index_range(
        subtask_index=config.subtask_index,
        subtask_end_index=getattr(config, "subtask_end_index", None),
    )

    logger.info(
        f"Subtask eval mode: {len(episodes_to_run)} episodes, "
        f"skill_filter={skill_filter}, subtask_range={selected_subtask_range}"
    )
    logger.info("Subtask success source: task-specific reward stage completion")

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    summary_results = []

    with Evaluator(config) as evaluator:
        logger.info("Starting subtask evaluation...")

        for episode_index in episodes_to_run:
            instance_id = int((episode_index // 10) % 1e3)
            ep_dir = orchestrator_dir / f"episode_{episode_index:08d}"

            subtask_files = sorted(
                ep_dir.glob("subtask_*_annotated.json"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            subtask_paths = {int(path.stem.split("_")[1]): path for path in subtask_files}
            if selected_subtask_range is None:
                subtask_eval_ranges = [(idx, idx) for idx in sorted(subtask_paths)]
            else:
                range_start_idx, range_end_idx = selected_subtask_range
                missing_subtasks = [idx for idx in range(range_start_idx, range_end_idx + 1) if idx not in subtask_paths]
                assert not missing_subtasks, (
                    f"Episode {episode_index} is missing requested subtasks {missing_subtasks} under {ep_dir}"
                )
                subtask_eval_ranges = [selected_subtask_range]

            for subtask_start_idx, subtask_end_idx in subtask_eval_ranges:
                selected_subtask_infos = []
                for subtask_idx in range(subtask_start_idx, subtask_end_idx + 1):
                    subtask_file = subtask_paths[subtask_idx]
                    with open(subtask_file) as f:
                        selected_subtask_infos.append((subtask_idx, json.load(f)))

                start_frame, end_frame = resolve_subtask_frame_range(
                    demo_data_dir=demo_data_dir,
                    task_index=task_idx,
                    episode_index=episode_index,
                    subtask_index=subtask_start_idx,
                    subtask_end_index=subtask_end_idx,
                )

                selected_skill_descriptions = [
                    info["skill_description"] for _, info in selected_subtask_infos
                ]
                if skill_filter is not None and not all(skill in skill_filter for skill in selected_skill_descriptions):
                    continue

                start_subtask_info = selected_subtask_infos[0][1]
                end_subtask_info = selected_subtask_infos[-1][1]
                subtask_desc = (
                    start_subtask_info["cot_subtask_description"]
                    if subtask_start_idx == subtask_end_idx
                    else " -> ".join(info["cot_subtask_description"] for _, info in selected_subtask_infos)
                )
                skill_desc = (
                    start_subtask_info["skill_description"]
                    if subtask_start_idx == subtask_end_idx
                    else " -> ".join(selected_skill_descriptions)
                )
                subtask_label = _format_subtask_range_label(subtask_start_idx, subtask_end_idx)

                logger.info("")
                logger.info("=" * 60)
                logger.info(
                    f"Episode {episode_index} | Subtask {subtask_label} | {subtask_desc}"
                )
                logger.info(
                    f"Skills: {skill_desc} | Frames: {start_frame}-{end_frame} | Instance: {instance_id}"
                )
                logger.info("=" * 60)

                evaluator.reset()
                evaluator.load_task_instance(instance_id)
                sync_task_reward_annotation_for_episode(
                    task=evaluator.env.task,
                    demo_data_dir=demo_data_dir,
                    task_index=task_idx,
                    episode_index=episode_index,
                    logger=logger,
                )
                evaluator.reset()
                evaluator.load_subtask_init_state(
                    demo_data_dir, task_idx, episode_index, start_frame,
                )

                start_stage_name = _resolve_subtask_reward_stage_name(
                    evaluator=evaluator,
                    subtask_idx=subtask_start_idx,
                )
                target_stage_name = _resolve_subtask_reward_stage_name(
                    evaluator=evaluator,
                    subtask_idx=subtask_end_idx,
                )
                _prime_task_reward_for_subtask(evaluator=evaluator, subtask_idx=subtask_start_idx)
                logger.info(
                    f"Reward stage window: {start_stage_name} -> {target_stage_name}"
                )

                subtask_duration = end_frame - start_frame
                subtask_max_steps = int(subtask_duration * config.subtask_max_steps_multiplier)
                if config.max_steps is not None:
                    subtask_max_steps = config.max_steps
                logger.info(f"Max steps: {subtask_max_steps} (duration={subtask_duration})")

                done = False
                step_count = 0
                reward_stage_success = False
                reached_target_stage = False
                reward_stage_info = {}
                reward_info = {}
                if config.write_video:
                    video_name = (
                        str(video_path)
                        + f"/{config.task.name}_ep{episode_index}_st{subtask_start_idx}"
                        + ("" if subtask_start_idx == subtask_end_idx else f"_to_{subtask_end_idx}")
                        + f"_{start_subtask_info['skill_description'].replace(' ', '_')}.mp4"
                    )
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )

                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                while not done:
                    terminated, truncated, reward, info = evaluator.step()
                    step_count += 1

                    reward_stage_success, reward_stage_info, reward_info = _get_reward_stage_result(
                        info=info,
                        target_stage_name=target_stage_name,
                    )
                    reached_target_stage = reached_target_stage or reward_stage_success

                    if (
                        terminated
                        or truncated
                        or step_count >= subtask_max_steps
                        or (reward_stage_success and not config.keep_running_after_success)
                        or evaluator.last_policy_done
                    ):
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if step_count % 100 == 0:
                        logger.info(
                            f"  step={step_count}, bddl_reward={reward:.4f}, "
                            f"reward_stage={reward_stage_info}"
                        )
                        for line in summarize_stage_progress(info):
                            logger.info(f"  {line}")

                if config.write_video and terminated:
                    for _ in range(3):
                        obs, _, _, _, _ = evaluator.env.step(
                            evaluator.robot_action, n_render_iterations=3
                        )
                        evaluator.obs = evaluator._preprocess_obs(obs)
                        evaluator._write_video()

                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)

                logger.info(
                    f"Finished: steps={step_count}, terminated={terminated}, truncated={truncated}"
                )
                logger.info(f"Reward stage result: {target_stage_name} -> {reward_stage_info}")
                if evaluator.last_policy_done:
                    logger.info("Policy server reported done at the end of the selected subtask clip/range.")

                subtask_result = {
                    "episode_index": episode_index,
                    "subtask_idx": subtask_start_idx,
                    "subtask_start_idx": subtask_start_idx,
                    "subtask_end_idx": subtask_end_idx,
                    "subtask_range_label": subtask_label,
                    "skill_description": skill_desc,
                    "skill_descriptions": selected_skill_descriptions,
                    "cot_subtask_description": subtask_desc,
                    "cot_subtask_descriptions": [
                        info["cot_subtask_description"] for _, info in selected_subtask_infos
                    ],
                    "manipulating_object_id": end_subtask_info.get("manipulating_object_id", []),
                    "steps": step_count,
                    "subtask_success": reached_target_stage,
                    "reward_start_stage_name": start_stage_name,
                    "reward_stage_name": target_stage_name,
                    "reward_stage_info": reward_stage_info,
                    "reward_current_stage_name": reward_info.get("current_stage_name"),
                    "reward_completed_stage_count": reward_info.get("completed_stage_count"),
                    "reward_total_stage_count": reward_info.get("total_stage_count"),
                }
                summary_results.append(subtask_result)

                metrics_suffix = (
                    f"st{subtask_start_idx}"
                    if subtask_start_idx == subtask_end_idx
                    else f"st{subtask_start_idx}_to_{subtask_end_idx}"
                )
                metrics_file = metrics_path / f"{config.task.name}_ep{episode_index}_{metrics_suffix}.json"
                with open(metrics_file, "w") as f:
                    json.dump(subtask_result, f, indent=2)

                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video: {video_name}")

    # --- Aggregated summary ---
    n_total = len(summary_results)
    n_success = sum(1 for r in summary_results if r["subtask_success"])
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUBTASK EVAL SUMMARY: {n_success}/{n_total} succeeded ({100*n_success/max(n_total,1):.1f}%)")
    logger.info("=" * 60)

    summary_path = metrics_path / f"{config.task.name}_subtask_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"success_rate": n_success / max(n_total, 1), "results": summary_results}, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")


def _run_instance_eval(config, logger):
    """Original instance-level evaluation (unchanged logic)."""
    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")
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

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = str(video_path) + f"/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated, reward, info = evaluator.step()
                    if terminated or truncated or evaluator.last_policy_done:
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 100 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                        logger.info(f"Current reward: {reward}")
                        logger.info(f"Current info: {info}")
                        for line in summarize_stage_progress(info):
                            logger.info(line)

                if config.write_video and terminated:
                    for _ in range(3):
                        obs, _, _, _, _ = evaluator.env.step(
                            evaluator.robot_action, n_render_iterations=3
                        )
                        evaluator.obs = evaluator._preprocess_obs(obs)
                        evaluator._write_video()

                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                if evaluator.last_policy_done:
                    logger.info("Policy server reported done and ended the replay clip.")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")


if __name__ == "__main__":
    register_omegaconf_resolvers()
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda:0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    gm.HEADLESS = config.headless

    assert config.eval_level in {"instance", "subtask"}, (
        f"Unsupported eval_level '{config.eval_level}'. Expected 'instance' or 'subtask'."
    )

    if config.eval_level == "subtask":
        _run_subtask_eval(config, logger)
    else:
        _run_instance_eval(config, logger)
