from copy import deepcopy
from typing import Dict, List


def extract_sequential_reward_info(info: Dict) -> Dict:
    """
    Get the sequential-task reward payload from the environment step info.
    """
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
    """
    Keep rollout alive after an early BDDL success.

    Before all sequential stages complete, suppress success so evaluation can
    continue to the final stage. After all stages complete, preserve success but
    still request continued rollout so callers can observe post-success behavior
    until timeout / max steps.
    """
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


def _get_active_stage_payload(info: Dict) -> tuple[str | None, Dict]:
    reward_info = extract_sequential_reward_info(info)
    current_stage_name = reward_info.get("current_stage_name")
    stage_infos = reward_info.get("stage_infos")
    if not isinstance(current_stage_name, str) or not isinstance(stage_infos, dict):
        return None, {}
    active_stage_info = stage_infos.get(current_stage_name)
    if not isinstance(active_stage_info, dict):
        return current_stage_name, {}
    return current_stage_name, active_stage_info


def _format_active_stage_lines(info: Dict) -> List[str]:
    reward_info = extract_sequential_reward_info(info)
    current_stage_name, active_stage_info = _get_active_stage_payload(reward_info)
    if current_stage_name is None:
        return []

    lines = []
    stage_rewards = reward_info.get("stage_rewards")
    if isinstance(stage_rewards, dict) and current_stage_name in stage_rewards:
        lines.append(f"active_stage_reward: {current_stage_name}={_format_scalar(stage_rewards[current_stage_name])}")

    condition_parts = []
    metric_parts = []
    for key, value in active_stage_info.items():
        if key == "reward":
            continue
        if isinstance(value, bool):
            condition_parts.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            metric_parts.append(f"{key}={_format_scalar(value)}")
        else:
            condition_parts.append(f"{key}={value}")

    if condition_parts:
        lines.append("active_conditions: " + ", ".join(condition_parts))
    if metric_parts:
        lines.append("active_metrics: " + ", ".join(metric_parts))

    return lines


def format_stage_status_chain(info: Dict) -> str | None:
    """
    Convert sequential reward info into a compact ordered stage chain.
    """
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


def format_video_info_lines(info: Dict, step: int, reward: float) -> List[str]:
    """
    Build the task-focused banner lines shown above the video.
    """
    raw_info = info or {}
    reward_info = extract_sequential_reward_info(raw_info)
    lines = [f"step={step} reward={reward:.4f}"]

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

    lines.extend(_format_active_stage_lines(reward_info))

    done_info = raw_info.get("done")
    if isinstance(done_info, dict) and "success" in done_info:
        lines.append(f"task_done: success={done_info['success']}")

    return lines


def summarize_stage_progress(info: Dict) -> List[str]:
    """
    Build logger-friendly task progress lines.
    """
    raw_info = info or {}
    reward_info = extract_sequential_reward_info(raw_info)
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

    lines.extend(_format_active_stage_lines(reward_info))

    done_info = raw_info.get("done")
    if isinstance(done_info, dict) and "success" in done_info:
        lines.append(f"task_done: success={done_info['success']}")

    return lines
