import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
from gello.robots.sim_robot.og_teleop_utils import (
    load_available_tasks,
    generate_robot_config,
)
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.demo_expert_policy_server import ParquetDemoReplayPolicy
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    build_subtask_eval_targets,
    delay_termination_until_stage_completion,
    format_subtask_range_label,
    generate_basic_environment_config,
    flatten_obs_dict,
    get_task_specific_reward,
    get_reward_stage_result,
    format_stage_progress_lines,
    get_instance_to_run,
    resolve_episode_indices,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm, create_module_macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Optional, Tuple, List

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_EVAL_INSTANCES = 10


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

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
            task_reward_kwargs["orchestrators_annotation_dir"] = self.cfg.orchestrators_annotation_dir
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
        logger.info(
            f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
        )
        cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
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

    def _query_policy_action(self, policy: Any, obs: dict) -> th.Tensor:
        """
        Query either a websocket-style policy or a local replay policy using one shared adapter.
        """
        if hasattr(policy, "forward"):
            action = policy.forward(obs=obs)
        elif hasattr(policy, "act"):
            action = policy.act(obs)
        else:
            raise AttributeError(f"Unsupported policy interface: {type(policy)}")

        # Convert local numpy demo actions into the tensor format expected by the environment.
        return th.as_tensor(action, dtype=th.float32)

    def _policy_needs_obs(self, policy: Any) -> bool:
        """
        Return whether the next policy query requires a fresh observation.
        """
        return bool(getattr(policy, "needs_obs", True))

    def _step_with_policy(
        self,
        policy: Any,
        *,
        apply_success_delay: bool,
        run_metrics: bool,
        track_trial_results: bool,
    ) -> Tuple[bool, bool, float, dict]:
        """
        Step the environment with an arbitrary policy while controlling evaluation bookkeeping.
        """
        current_obs = self.obs if self._policy_needs_obs(policy) else None
        self.robot_action = self._query_policy_action(policy, current_obs)
        needs_obs_after_step = self._policy_needs_obs(policy)

        if needs_obs_after_step or self.cfg.write_video:
            obs, reward, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
            self.obs = self._preprocess_obs(obs)
        else:
            # Skip rendering and camera reads for cached VLA chunk actions.
            with og.sim.render_on_step(False):
                terminated, truncated, reward, info = self._fast_env_step(self.robot_action)
        if terminated and not truncated and apply_success_delay and (
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
        self.last_step_reward = reward
        self.last_step_info = info
        self.last_policy_done = bool(getattr(policy, "is_done", False))

        if track_trial_results and (terminated or truncated):
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        if run_metrics:
            for metric in self.metrics:
                metric.step_callback(self.env)
        return terminated, truncated, reward, info

    def _fast_env_step(self, action: th.Tensor) -> Tuple[bool, bool, float, dict]:
        """
        Step the simulator without collecting observations.
        """
        action = self.env._convert_action_to_tensor(action)
        self.env._pre_step(action)
        og.sim.step()

        # Skip get_obs() entirely to make warmup as light as possible.
        reward, done, info = self.env.task.step(self.env, action)
        self.env._populate_info(info)

        terminated = False
        truncated = False
        for tc, tc_data in info["done"]["termination_conditions"].items():
            if tc_data["done"]:
                if tc == "timeout":
                    truncated = True
                else:
                    terminated = True
        assert (terminated or truncated) == done, "Terminated and truncated must match done!"

        self.last_step_reward = reward
        self.last_step_info = info
        self.env._current_step += 1
        return terminated, truncated, reward, info

    def _fast_step_with_policy(self, policy: Any) -> Tuple[bool, bool, float, dict]:
        """
        Step the simulator without collecting observations after querying a policy.
        """
        current_obs = self.obs if self._policy_needs_obs(policy) else None
        self.robot_action = self._query_policy_action(policy, current_obs)
        terminated, truncated, reward, info = self._fast_env_step(self.robot_action)
        self.last_policy_done = bool(getattr(policy, "is_done", False))
        return terminated, truncated, reward, info

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
        return self._step_with_policy(
            self.policy,
            apply_success_delay=True,
            run_metrics=True,
            track_trial_results=True,
        )

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

    def load_task_instance(self, instance_id: int) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
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

    def update_orchestrators_annotation_dir(self, orchestrators_annotation_dir: Path) -> None:
        """
        Update the active orchestrator annotation directory before any environment reset.
        """
        self.cfg["orchestrators_annotation_dir"] = orchestrators_annotation_dir
        task_reward = get_task_specific_reward(self)
        if hasattr(task_reward, "orchestrators_annotation_dir"):
            # Keep the reward function pointed at the current episode annotations.
            task_reward.orchestrators_annotation_dir = orchestrators_annotation_dir

    def replay_demo_to_preparatory_state(
        self,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        target_start_frame: int,
        required_warmup_stage_idx: Optional[int] = None,
    ) -> dict:
        """
        Replay demo actions from frame 0 until the target subtask start frame.
        """
        if target_start_frame <= 0:
            logger.info("Skipping demo warmup because the target subtask starts at frame 0.")
            return {
                "used_demo_warmup": False,
                "warmup_steps": 0,
                "warmup_end_frame": 0,
                "warmup_ready_for_policy": True,
            }

        parquet_path = (
            Path(demo_data_dir) / "data" / f"task-{task_index:04d}" / f"episode_{episode_index:08d}.parquet"
        )
        replay_policy = ParquetDemoReplayPolicy(
            parquet_path=parquet_path,
            start_frame=0,
            end_frame=target_start_frame,
            subtask_index=0,
            subtask_end_index=0,
        )
        replay_policy.reset()

        logger.info(
            "Replaying demo warmup for episode=%d until preparatory frame=%d using %s",
            episode_index,
            target_start_frame,
            parquet_path,
        )

        warmup_steps = 0
        warmup_ready_for_policy = required_warmup_stage_idx is None
        # Disable render-on-step during warmup because the replay policy does not consume images.
        with og.sim.render_on_step(False):
            while not replay_policy.is_done:
                terminated, truncated, reward, info = self._fast_step_with_policy(replay_policy)
                if required_warmup_stage_idx is not None:
                    # Track whether the last prerequisite stage has become completed by the current warmup step.
                    warmup_ready_for_policy, _, _ = get_reward_stage_result(
                        info=info,
                        target_stage_idx=required_warmup_stage_idx,
                    )
                warmup_steps += 1
                if warmup_steps % 100 == 0 or warmup_ready_for_policy==True:
                    logger.info(f"  warmup_step={warmup_steps}/{target_start_frame} reward={reward:.4f} warmup_ready_for_policy={warmup_ready_for_policy}")

        # Render only once at the handoff boundary so the first VLA step sees fresh camera observations.
        og.sim.render()
        self.obs = self._preprocess_obs(self.env.get_obs()[0])
        self.last_policy_done = False
        logger.info(
            "Reached preparatory state after %d warmup steps for episode=%d.",
            warmup_steps,
            episode_index,
        )
        return {
            "used_demo_warmup": True,
            "warmup_steps": warmup_steps,
            "warmup_end_frame": target_start_frame,
            "warmup_ready_for_policy": warmup_ready_for_policy,
        }

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
        return obs

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        frame = self._compose_video_frame()
        if frame is None:
            return
        write_video(
            np.expand_dims(frame, 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _compose_video_frame(self) -> Optional[np.ndarray]:
        """
        Compose a fixed-size video frame so camera views keep their intended aspect ratio
        even when the info banner is present.
        """
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return None

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
        return overlay_info_banner(
            frame,
            info=self.last_step_info,
            step=self.env._current_step,
            reward=self.last_step_reward,
            banner_height=256,
        )

    def _get_video_resolution(self) -> Tuple[int, int]:
        """
        Return the final video resolution after the banner is added.
        """
        frame = self._compose_video_frame()
        assert frame is not None, "Cannot infer video resolution without RGB camera observations."
        return frame.shape[:2]

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
        # print stats
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


def _run_subtask_eval(config, logger):
    """Subtask-level evaluation: iterate episodes × subtasks from demo data."""
    task_idx = TASK_NAMES_TO_INDICES[config.task.name]
    assert config.demo_data_dir is not None, "demo_data_dir must be set when eval_level=subtask."
    episode_indices = resolve_episode_indices(
        config.demo_data_dir,
        task_idx,
        run_episode_idx=config.run_episode_idx,
        run_episode_indices=config.get("run_episode_indices"),
    )
    assert len(episode_indices) > 0, (
        f"No episode indices resolved for task={config.task.name} under {config.demo_data_dir}"
    )

    logger.info(
        "Subtask eval mode: episodes=%s, subtask_skill=%s, subtask_range=(%s, %s)",
        episode_indices,
        config.get("subtask_skill"),
        config.subtask_index,
        config.subtask_end_index,
    )
    logger.info("Subtask success source: task-specific reward stage completion")

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    summary_results = []
    initial_orchestrators_annotation_dir = (
        Path(config.demo_data_dir) / "orchestrators" / f"task-{task_idx:04d}" / f"episode_{episode_indices[0]:08d}"
    )
    logger.info("Initial orchestrators annotation directory: %s", initial_orchestrators_annotation_dir)
    config["orchestrators_annotation_dir"] = initial_orchestrators_annotation_dir
    
    with Evaluator(config) as evaluator:
        logger.info("Starting subtask evaluation...")
        for episode_index in episode_indices:
            skip_episode = False
            orchestrators_annotation_dir = (
                Path(config.demo_data_dir) / "orchestrators" / f"task-{task_idx:04d}" / f"episode_{episode_index:08d}"
            )
            logger.info("Orchestrators annotation directory: %s", orchestrators_annotation_dir)
            evaluator.update_orchestrators_annotation_dir(orchestrators_annotation_dir)

            eval_targets = build_subtask_eval_targets(
                orchestrators_annotation_dir,
                subtask_skill=config.get("subtask_skill"),
                subtask_index=config.subtask_index,
                subtask_end_index=config.subtask_end_index,
            )
            instance_id = int((episode_index // 10) % 1e3)

            for eval_target in eval_targets:
                selected_subtask_infos = eval_target["selected_subtask_infos"]
                subtask_start_idx = eval_target["subtask_start_idx"]
                subtask_end_idx = eval_target["subtask_end_idx"]

                start_subtask_info = selected_subtask_infos[0][1]
                end_subtask_info = selected_subtask_infos[-1][1]
                start_frame = int(start_subtask_info["start_frame"])
                end_frame = int(end_subtask_info["end_frame"])
                assert start_frame < end_frame, (
                    f"Subtask selection task={task_idx}, episode={episode_index}, subtasks=[{subtask_start_idx}, {subtask_end_idx}] "
                    f"has invalid frame range: start_frame={start_frame}, end_frame={end_frame}"
                )

                selected_skill_descriptions = [
                    info["skill_description"] for _, info in selected_subtask_infos
                ]
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
                subtask_label = format_subtask_range_label(subtask_start_idx, subtask_end_idx)

                logger.info("")
                logger.info("=" * 60)
                logger.info(
                    "Episode %d | Subtask %s | %s",
                    episode_index,
                    subtask_label,
                    subtask_desc,
                )
                logger.info(
                    "Skills: %s | Frames: %d-%d | Instance: %d | Selection: %s=%s",
                    skill_desc,
                    start_frame,
                    end_frame,
                    instance_id,
                    eval_target["selection_mode"],
                    eval_target["selection_value"],
                )
                logger.info("=" * 60)

                evaluator.reset()
                evaluator.load_task_instance(instance_id)
                evaluator.reset()
                warmup_result = evaluator.replay_demo_to_preparatory_state(
                    demo_data_dir=config.demo_data_dir,
                    task_index=task_idx,
                    episode_index=episode_index,
                    target_start_frame=start_frame,
                    required_warmup_stage_idx=(subtask_start_idx - 1) if subtask_start_idx > 0 else None,
                )
                if not warmup_result["warmup_ready_for_policy"]:
                    logger.warning(
                        "Skipping episode %d because warmup did not complete prerequisite stage %d before subtask %d.",
                        episode_index,
                        subtask_start_idx - 1,
                        subtask_start_idx,
                    )
                    skip_episode = True
                    break

                # Reset the VLA policy only after demo warmup so the first inference sees the preparatory state.
                evaluator.policy.reset()
                evaluator.last_policy_done = False

                subtask_duration = end_frame - start_frame
                subtask_max_steps = int(subtask_duration * config.subtask_max_steps_multiplier)
                if config.max_steps is not None:
                    subtask_max_steps = config.max_steps
                logger.info(f"Max steps: {subtask_max_steps} (duration={subtask_duration})")

                done = False
                terminated = False
                truncated = False
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
                        resolution=evaluator._get_video_resolution(),
                    )

                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                while not done:
                    terminated, truncated, reward, info = evaluator.step()
                    step_count += 1

                    reward_stage_success, reward_stage_info, reward_info = get_reward_stage_result(
                        info=info,
                        target_stage_idx=subtask_end_idx,
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
                        for line in format_stage_progress_lines(info):
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
                logger.info(f"Reward stage result: {subtask_end_idx} -> {reward_stage_info}")
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
                    "reward_stage_info": reward_stage_info,
                    "reward_current_stage_name": reward_info.get("current_stage_name"),
                    "reward_completed_stage_count": reward_info.get("completed_stage_count"),
                    "reward_total_stage_count": reward_info.get("total_stage_count"),
                    "selection_mode": eval_target["selection_mode"],
                    "selection_value": eval_target["selection_value"],
                    "used_demo_warmup": warmup_result["used_demo_warmup"],
                    "warmup_steps": warmup_result["warmup_steps"],
                    "warmup_end_frame": warmup_result["warmup_end_frame"],
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
            if skip_episode:
                continue

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
    
    instances_to_run = get_instance_to_run(config, m, gm, logger)

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    task_idx = TASK_NAMES_TO_INDICES[config.task.name]

    episode_index = config.run_episode_idx
    orchestrators_annotation_dir = Path(config.demo_data_dir) / "orchestrators" / f"task-{task_idx:04d}" / f"episode_{episode_index:08d}"
    logger.info(f"Orchestrators annotation directory: {orchestrators_annotation_dir}")
    config["orchestrators_annotation_dir"] = orchestrators_annotation_dir
    
    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.load_task_instance(idx)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = str(video_path) + f"/video_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=evaluator._get_video_resolution(),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated, reward, info = evaluator.step()
                    if terminated or truncated or evaluator.last_policy_done:
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                        logger.info(f"Current reward: {reward}")
                        logger.info(f"Current info: {info}")
                        for line in format_stage_progress_lines(info):
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
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}::{idx}::{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
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
