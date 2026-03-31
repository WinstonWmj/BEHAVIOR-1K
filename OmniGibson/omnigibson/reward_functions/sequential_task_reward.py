import math
from abc import abstractmethod
from copy import deepcopy

import torch as th

from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class SequentialTaskReward(BaseRewardFunction):
    """
    Base reward for task-bound, sequential subtasks.

    Each stage is evaluated strictly in order. Only the active stage contributes
    dense reward, and stage transition happens only after the active stage is
    marked completed.
    """

    def __init__(self, stage_completion_bonus=1.0):
        self.stage_completion_bonus = stage_completion_bonus
        self._stage_index = 0
        self._stage_defs = []
        self._completed_stage_names = set()
        self._total_stages = 0
        self._stage_cumulative_rewards = {}
        super().__init__()

    def reset(self, task, env):
        super().reset(task, env)
        self._stage_defs = self._build_stages(task=task, env=env)
        self._stage_index = 0
        self._completed_stage_names = set()
        self._total_stages = len(self._stage_defs)
        self._stage_cumulative_rewards = {stage["name"]: 0.0 for stage in self._stage_defs}
        for stage in self._stage_defs:
            stage.setdefault("state", {})

    def set_active_stage_index(self, stage_index: int) -> None:
        """
        Prime the sequential reward to treat all stages before `stage_index`
        as already completed, making `stage_index` the current active stage.
        """
        assert 0 <= stage_index < self._total_stages, (
            f"stage_index must be in [0, {self._total_stages - 1}], got {stage_index}"
        )
        self._stage_index = int(stage_index)
        completed_stages = self._stage_defs[:stage_index]
        self._completed_stage_names = {stage["name"] for stage in completed_stages}
        for idx, stage in enumerate(self._stage_defs):
            self._stage_cumulative_rewards[stage["name"]] = self.stage_completion_bonus if idx < stage_index else 0.0

    @abstractmethod
    def _build_stages(self, task, env):
        raise NotImplementedError()

    @abstractmethod
    def _evaluate_stage(self, stage, task, env, action):
        """
        Returns:
            dict with keys:
                reward: float
                completed: bool
                metrics: dict
        """
        raise NotImplementedError()

    def _step(self, task, env, action):
        stage_rewards = {stage["name"]: 0.0 for stage in self._stage_defs}
        stage_infos = {
            stage["name"]: {
                "completed": stage["name"] in self._completed_stage_names,
                "reward": 0.0,
            }
            for stage in self._stage_defs
        }
        total_reward = 0.0

        while self._stage_index < self._total_stages:
            active_stage = self._stage_defs[self._stage_index]
            stage_name = active_stage["name"]
            result = self._evaluate_stage(stage=active_stage, task=task, env=env, action=action)

            stage_reward = float(result.get("reward", 0.0))
            stage_completed = bool(result.get("completed", False))
            stage_metrics = deepcopy(result.get("metrics", {}))

            stage_rewards[stage_name] = stage_reward
            stage_infos[stage_name] = {
                "completed": stage_completed,
                "reward": stage_reward,
                **stage_metrics,
            }
            self._stage_cumulative_rewards[stage_name] += stage_reward
            total_reward += stage_reward

            if not stage_completed:
                break

            if stage_name not in self._completed_stage_names:
                total_reward += self.stage_completion_bonus
                stage_rewards[stage_name] += self.stage_completion_bonus
                stage_infos[stage_name]["reward"] += self.stage_completion_bonus
                stage_infos[stage_name]["completion_bonus"] = self.stage_completion_bonus
                self._stage_cumulative_rewards[stage_name] += self.stage_completion_bonus
                self._completed_stage_names.add(stage_name)

            self._stage_index += 1

        current_stage_name = (
            self._stage_defs[self._stage_index]["name"] if self._stage_index < self._total_stages else "done"
        )
        info = {
            "current_stage_idx": min(self._stage_index, self._total_stages),
            "current_stage_name": current_stage_name,
            "completed_stage_count": len(self._completed_stage_names),
            "total_stage_count": self._total_stages,
            "all_stages_completed": self._stage_index >= self._total_stages,
            "stage_rewards": stage_rewards,
            "stage_cumulative_rewards": deepcopy(self._stage_cumulative_rewards),
            "stage_infos": stage_infos,
        }
        return total_reward, info

    @staticmethod
    def _progress_reward(previous_value, current_value, scale, invert=False):
        if previous_value is None:
            return 0.0
        delta = (previous_value - current_value) if invert else (current_value - previous_value)
        return float(delta) * scale

    @staticmethod
    def _exp_distance_reward(distance, scale):
        return math.exp(-max(distance, 0.0)) * scale
