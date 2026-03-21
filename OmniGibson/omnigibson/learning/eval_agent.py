import csv
import base64
import cv2
import hydra
import importlib
import json
import logging
import mimetypes
import time
import datetime
import re
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
    generate_basic_environment_config,
    flatten_obs_dict,
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
from typing import Any, Tuple, List, Optional, Dict

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10
m.STEP_TIMEOUT_SEC = 15 * 60
m.SLOW_STEP_WARN_SEC = 2.0

DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
STEP_IMAGE_CHECK_PROMPT = "请判断这张图里机器人是否打开了柜门，并简要说明依据。"
STEP_POLICY_DECISION_PROMPT = (
    "请根据视觉判断结果决定控制动作："
    "如果确认打开了柜门，优先切换到更合适的policy；若无需切换则继续；若必须立即结束则停止。"
)


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
        self.policy_endpoint_registry = self._build_endpoint_registry()
        self.current_policy_name = self.cfg.policy_name
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None
        self.inspection_path = Path(self.cfg.log_path).expanduser() / "step_inspections"
        self.inspection_path.mkdir(parents=True, exist_ok=True)

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
        logger.info(f"Loaded policy: {self.current_policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def _build_endpoint_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a name -> {host, port} registry for websocket policy switching.

        The current policy is auto-registered. Additional endpoints come from
        ``policy_endpoints`` in config, e.g. CLI:
            +policy_endpoints.openvla.host=localhost +policy_endpoints.openvla.port=8001
        """
        current_host = self.cfg.model.get("host", "localhost")
        current_port = self.cfg.model.get("port", 8000)
        registry: Dict[str, Dict[str, Any]] = {
            self.cfg.policy_name: {"host": str(current_host), "port": int(current_port)},
        }
        endpoints = self.cfg.get("policy_endpoints", None)
        if endpoints is not None:
            for name, ep in endpoints.items():
                registry[str(name)] = {"host": str(ep.get("host", "localhost")), "port": int(ep.get("port", 8000))}
        logger.info(
            "Policy endpoint registry: %s",
            {k: f"{v['host']}:{v['port']}" for k, v in registry.items()},
        )
        return registry

    def switch_policy(self, policy_name: str) -> bool:
        """
        Switch to another pre-running policy server by redirecting the websocket connection.
        The target server must already be running on the registered host:port.
        """
        target = policy_name.strip()
        if target == self.current_policy_name:
            logger.info("Policy switch skipped: already using '%s'.", target)
            return True
        if target not in self.policy_endpoint_registry:
            logger.warning(
                "Policy switch failed: '%s' not in registry. Available: %s",
                target,
                sorted(self.policy_endpoint_registry.keys()),
            )
            return False

        ep = self.policy_endpoint_registry[target]
        old_name = self.current_policy_name
        try:
            from omnigibson.learning.policies import WebsocketPolicy
            if isinstance(self.policy, WebsocketPolicy):
                self.policy.update_host(ep["host"], ep["port"])
            else:
                logger.info("Current policy is not WebsocketPolicy, creating new WebsocketPolicy for '%s'.", target)
                self.policy = WebsocketPolicy(host=ep["host"], port=ep["port"])
            self.current_policy_name = target
            self.policy.reset()
        except Exception:
            logger.exception("Failed to switch policy to '%s'. Keep current '%s'.", target, old_name)
            return False

        logger.info("Policy switched: %s -> %s (%s:%d)", old_name, target, ep["host"], ep["port"])
        return True

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool, dict]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.

        Returns:
            Tuple[bool, bool, dict]:
                - terminated (bool): Whether the episode has terminated (i.e., reached a terminal state).
                - truncated (bool): Whether the episode was truncated (i.e., stopped due to a time limit or other constraint).
                - info (dict): Environment info with added timing fields.

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
        forward_start = time.time()
        self.robot_action = self.policy.forward(obs=self.obs)
        policy_forward_time = time.time() - forward_start

        env_step_start = time.time()
        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        env_step_time = time.time() - env_step_start
        # process obs
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        info["policy_forward_time"] = policy_forward_time
        info["env_step_time"] = env_step_time
        return terminated, truncated, info

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
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    @staticmethod
    def _to_image_url(image: str) -> str:
        """
        将图片输入转换成 OpenAI 兼容的 image_url。
        - 若是 http/https URL：原样返回
        - 若是本地路径：转为 data URL
        """
        if image.startswith(("http://", "https://")):
            return image

        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    @staticmethod
    def _call_dashscope_chat(
        instruction: str,
        image: str = "",
        *,
        model: str = "qwen-vl-max-latest",
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout: float = 60.0,
    ) -> str:
        """
        输入图片 + 指令，返回模型文本结果（兼容纯文本输入）。
        """
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"未找到 API Key。请先设置环境变量 {api_key_env}，例如:\n"
                f"export {api_key_env}='你的key'"
            )

        try:
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")
        except Exception as exc:
            raise ImportError(
                "缺少依赖 openai，请先安装: pip install openai"
            ) from exc

        client: Any = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL, timeout=timeout)
        content = [{"type": "text", "text": instruction}]
        if image:
            image_url = Evaluator._to_image_url(image)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, list):
            texts = []
            for item in message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(t for t in texts if t).strip()
        return str(message_content).strip()

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[dict]:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def inspect_step_images_and_decide(self, current_step: int) -> Dict[str, str]:
        """
        每 100 步触发一次：
        1) 导出三路图像；
        2) 按顺序逐张调用视觉模型判断；
        3) 将逐张返回文本再发送给大模型，判断继续/停止/切换策略。
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.inspection_path / f"step_{current_step}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

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

        image_entries = [
            ("01_left_wrist", left_wrist_rgb),
            ("02_right_wrist", right_wrist_rgb),
            ("03_head", head_rgb),
        ]

        per_image_results = []
        for name, image in image_entries:
            image_path = run_dir / f"{name}.png"
            # OpenCV 使用 BGR 顺序保存，观测一般为 RGB，这里转换后保存便于人工查看。
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            try:
                result = self._call_dashscope_chat(
                    instruction=STEP_IMAGE_CHECK_PROMPT,
                    image=str(image_path),
                    model="qwen3.5-plus",
                )
            except Exception as exc:
                result = f"[ERROR] {exc}"
                logger.exception("Step image inspection failed for %s at step %d", name, current_step)

            per_image_results.append((name, str(image_path), result))
            with open(run_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                f.write(result)

        combined_text = "\n\n".join(
            f"{idx}. {name}\nimage_path: {img_path}\nresult: {result}"
            for idx, (name, img_path, result) in enumerate(per_image_results, start=1)
        )

        decision_input = (
            f"这是按顺序（左腕、右腕、头部）得到的图像判断结果：\n\n{combined_text}\n\n"
            f"当前可切换policy列表：{', '.join(sorted(self.policy_endpoint_registry.keys()))}\n"
            "请严格输出 JSON，格式如下：\n"
            '{"action":"continue|stop|switch","policy_name":"<仅当action=switch时填写>","reason":"简短原因"}\n'
            "除了 JSON 不要输出任何其他内容。"
        )
        decision_prompt = f"{STEP_POLICY_DECISION_PROMPT}\n\n{decision_input}"
        try:
            decision_raw = self._call_dashscope_chat(
                instruction=decision_prompt,
                image="",
                model="qwen3.5-plus",
            )
        except Exception as exc:
            decision_raw = f"[ERROR] {exc}"
            logger.exception("Stop-decision API call failed at step %d", current_step)

        with open(run_dir / "decision_prompt.txt", "w", encoding="utf-8") as f:
            f.write(decision_prompt)
        with open(run_dir / "decision_result.txt", "w", encoding="utf-8") as f:
            f.write(decision_raw)

        parsed = self._extract_json_from_text(decision_raw)
        action = "continue"
        policy_name = ""
        if parsed is not None:
            action = str(parsed.get("action", "continue")).strip().lower()
            policy_name = str(parsed.get("policy_name", "")).strip()
        else:
            # 容错：模型未严格输出 JSON 时做关键词回退
            text = decision_raw.strip().lower()
            if "switch" in text or "切换" in decision_raw:
                action = "switch"
                for candidate in sorted(self.policy_endpoint_registry.keys(), key=len, reverse=True):
                    if candidate in decision_raw:
                        policy_name = candidate
                        break
            elif "stop" in text or "停止" in decision_raw:
                action = "stop"

        if action not in {"continue", "stop", "switch"}:
            action = "continue"
        if action == "switch" and not policy_name:
            logger.warning("Model requests switch but policy_name is empty, fallback to continue.")
            action = "continue"

        logger.info(
            "Step %d inspection saved to %s, action=%s, target_policy=%s, raw=%s",
            current_step,
            str(run_dir),
            action.upper(),
            policy_name,
            decision_raw,
        )
        return {"action": action, "policy_name": policy_name, "raw": decision_raw}

    def reset(self) -> None:
        """
        Reset the environment, policy, and compute metrics.
        """
        self.obs = self._preprocess_obs(self.env.reset()[0])
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        if hasattr(self.policy, "reset"):
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


def load_config_from_cli() -> DictConfig:
    """
    Register resolvers, compose Hydra config from CLI overrides, and apply global runtime flags.
    """
    register_omegaconf_resolvers()
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda:0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    gm.HEADLESS = config.headless
    return config


def setup_video_path(config: DictConfig) -> Optional[Path]:
    """
    Create and return the output video directory when video recording is enabled.
    """
    if not config.write_video:
        return None
    video_path = Path(config.log_path).expanduser() / "videos"
    video_path.mkdir(parents=True, exist_ok=True)
    return video_path


def setup_metrics_path(config: DictConfig) -> Path:
    """
    Create and return the output metrics directory.
    """
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    return metrics_path


def _resolve_eval_instance_ids(eval_instance_ids: Optional[List[int]], max_instances: int) -> List[int]:
    """
    Resolve eval instance IDs from config and validate they are in range.
    """
    resolved_ids = eval_instance_ids if eval_instance_ids is not None else list(range(max_instances))
    assert set(resolved_ids).issubset(
        set(range(max_instances))
    ), f"eval instance ids must be in range({max_instances})"
    return list(resolved_ids)


def select_instances_to_run(config: DictConfig) -> List[int]:
    """
    Select and validate task instances to evaluate according to runtime flags.
    """
    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

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
                instances_to_run.append(int((episode["episode_index"] // 10) % 1e3))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
        return instances_to_run

    eval_ids = _resolve_eval_instance_ids(config.eval_instance_ids, m.NUM_EVAL_INSTANCES)
    if config.test_hidden:
        return eval_ids

    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]
    assert (
        lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
    ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
    test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
    return [int(test_instances[i]) for i in eval_ids]


def check_step_timing(evaluator: Evaluator, policy_forward_time: float, env_step_time: float) -> None:
    """
    Validate step latency and emit timeout / slow-step diagnostics.
    """
    time_step = policy_forward_time + env_step_time
    if time_step > m.STEP_TIMEOUT_SEC:
        logger.error(f"Step timeout: {time_step} seconds, terminating evaluation")
        sys.exit(1)
    if time_step <= m.SLOW_STEP_WARN_SEC:
        return

    logger.warning(
        "Slow step: %.2fs at step %d (policy=%.2fs env+render=%.2fs). "
        "If policy >> env: policy server/GPU busy. If env >> policy: sim/GPU render slow.",
        time_step,
        evaluator.env._current_step,
        policy_forward_time,
        env_step_time,
    )


if __name__ == "__main__":
    config = load_config_from_cli()
    video_path = setup_video_path(config)
    instances_to_run = select_instances_to_run(config)
    metrics = {}
    metrics_path = setup_metrics_path(config)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if video_path is not None:
                    video_name = str(video_path) + f"/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated, info = evaluator.step()
                    check_step_timing(evaluator, info["policy_forward_time"], info["env_step_time"])
                    if terminated or truncated:
                        done = True
                    if video_path is not None:
                        evaluator._write_video()
                    if evaluator.env._current_step % 100 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                        decision = evaluator.inspect_step_images_and_decide(evaluator.env._current_step)
                        if decision["action"] == "switch":
                            switched = evaluator.switch_policy(decision["policy_name"])
                            if not switched:
                                logger.warning(
                                    "Switch action ignored due to failure, continue with policy '%s'.",
                                    evaluator.current_policy_name,
                                )
                        elif decision["action"] == "stop":
                            logger.warning("Model decision requests stopping current policy process.")
                            evaluator.video_writer = None
                            evaluator.env.close()
                            og.shutdown()
                            sys.exit(0)
                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
                if video_path is not None:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")
