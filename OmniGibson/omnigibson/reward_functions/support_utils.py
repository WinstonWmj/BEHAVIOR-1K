import json
import re

import torch as th

from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.touching import Touching
from omnigibson.utils.ui_utils import create_module_logger


log = create_module_logger(module_name=__name__)


def parse_support_label_from_annotation(annotation_path):
    if annotation_path is None:
        return None

    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _warn_exception("Annotation file read", exc, "Falling back to runtime support discovery.")
        return None

    for subtask_text in data.get("cot_subtask_description_list", []):
        lowered = str(subtask_text).lower().strip()
        match = re.search(r"(?:pick up|pickup|place)\s+.+?\s+(?:from|on)\s+(.+)", lowered)
        if match:
            return match.group(1).strip()

    return None


def find_support_object(task, env, target_obj, support_label=None):
    if target_obj is None:
        return None

    if support_label is not None:
        target_xy = None
        try:
            target_xy = get_obj_center(target_obj)[:2]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _warn_exception(
                "Target pose lookup for support matching",
                exc,
                "Skipping XY tie-breaking for support matching.",
            )

        normalized_label = _normalize_text(support_label)
        label_tokens = [tok for tok in normalized_label.split() if tok]
        for candidates in _iter_support_candidate_groups(task, env):
            best_obj = None
            best_score = -1
            best_xy_distance = float("inf")
            for scope_name, obj in candidates:
                if _should_skip_candidate(obj, target_obj):
                    continue

                candidate_text = " ".join(
                    _normalize_text(field)
                    for field in (
                        scope_name,
                        getattr(obj, "name", ""),
                        getattr(obj, "category", ""),
                        getattr(obj, "model", ""),
                        getattr(obj, "synset", ""),
                    )
                    if field
                )
                score = sum(token in candidate_text for token in label_tokens)
                if normalized_label and normalized_label in candidate_text:
                    score += 2

                xy_distance = float("inf")
                if target_xy is not None:
                    try:
                        xy_distance = float(th.norm(get_obj_center(obj)[:2] - target_xy).item())
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        _warn_exception(
                            "Candidate pose lookup for support ranking",
                            exc,
                            "Using infinite XY distance for this support candidate.",
                        )

                if score > best_score or (score == best_score and xy_distance < best_xy_distance):
                    best_obj = obj
                    best_score = score
                    best_xy_distance = xy_distance

            if best_score > 0:
                return best_obj

    for candidates, context in (
        (task.object_scope.values(), "task-scope"),
        (getattr(getattr(env, "scene", None), "objects", []), "scene"),
    ):
        for obj in candidates:
            if _should_skip_candidate(obj, target_obj):
                continue
            try:
                if OnTop in target_obj.states and target_obj.states[OnTop].get_value(obj):
                    return obj
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                _warn_exception(
                    f"OnTop query for {context} support candidate",
                    exc,
                    "Skipping this support candidate.",
                )

    return None


def infer_support_surface_height(obj):
    if obj is None:
        return None
    center, _, extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
    return (center[2] + extent[2] / 2.0).item()


def get_obj_center(obj):
    return obj.get_position_orientation()[0]


def get_min_eef_distance_to_obj(robot, obj):
    if obj is None:
        return float("inf")

    obj_pos = get_obj_center(obj)
    dists = []
    for arm in getattr(robot, "arm_names", []):
        try:
            eef_pos = robot.get_eef_position(arm)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            _warn_exception(
                f"End-effector pose lookup for arm {arm}",
                exc,
                "Ignoring this arm for object distance computation.",
            )
            continue
        dists.append(th.norm(eef_pos - obj_pos).item())

    if not dists:
        default_arm = getattr(robot, "default_arm", None)
        if default_arm is not None:
            eef_pos = robot.get_eef_position(default_arm)
            dists.append(th.norm(eef_pos - obj_pos).item())

    return min(dists) if dists else float("inf")


def get_min_eef_distance_to_toggle(robot, target_obj, toggle_state):
    if target_obj is None:
        return float("inf")

    if toggle_state is None or toggle_state.visual_marker is None:
        toggle_pos = get_obj_center(target_obj)
        marker_radius = 0.0
    else:
        toggle_pos = toggle_state.visual_marker.get_position_orientation()[0]
        marker_radius = th.min(toggle_state.visual_marker.extent * toggle_state.scale).item()

    dists = []
    for arm in getattr(robot, "arm_names", []):
        try:
            eef_pos = robot.get_eef_position(arm)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            _warn_exception(
                f"End-effector pose lookup for arm {arm}",
                exc,
                "Ignoring this arm for toggle distance computation.",
            )
            continue
        dists.append(th.norm(eef_pos - toggle_pos).item())

    if not dists:
        default_arm = getattr(robot, "default_arm", None)
        if default_arm is not None:
            eef_pos = robot.get_eef_position(default_arm)
            dists.append(th.norm(eef_pos - toggle_pos).item())

    raw_distance = min(dists) if dists else float("inf")
    return max(raw_distance - marker_radius, 0.0)


def is_target_in_hand(robot, target_obj):
    # This check does not infer grasp state from the current end-effector distance.
    # Instead, it reads robot._ag_obj_in_hand, which is the robot's own per-arm bookkeeping
    # for "which object is currently being held". In OmniGibson assisted / sticky grasping,
    # that field is populated when the robot finishes the internal grasp pipeline and creates
    # an assisted grasp joint, and it is cleared again on release.
    #
    # Therefore this function answers:
    # "Has the robot already registered the target object as being in hand?"
    # rather than:
    # "Is the gripper currently close enough to count as grasping?"
    #
    # Distance may still appear deeper in the robot grasp pipeline, but only as one heuristic
    # for selecting among already valid grasp candidates. For example, Robot._calculate_in_hand_object
    # first filters candidates by gripper contact (and ray-casting in assisted mode), and only then
    # uses distance to the gripper center to choose the closest candidate. That means proximity alone
    # does not make _ag_obj_in_hand become non-None.
    obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
    for arm in getattr(robot, "arm_names", []):
        if is_same_object(obj_in_hand.get(arm), target_obj):
            return True

    default_arm = getattr(robot, "default_arm", None)
    return default_arm is not None and is_same_object(obj_in_hand.get(default_arm), target_obj)


def is_same_object(obj_a, obj_b):
    if obj_a is None or obj_b is None:
        return False
    if obj_a is obj_b:
        return True
    for attr in ("prim_path", "name", "uuid"):
        value_a = getattr(obj_a, attr, None)
        value_b = getattr(obj_b, attr, None)
        if value_a is not None and value_b is not None and value_a == value_b:
            return True
    return False


def is_supported_by_surface(target_obj, support_obj, support_surface_height, support_contact_tolerance):
    if target_obj is None or support_obj is None:
        return False

    try:
        if OnTop in target_obj.states and target_obj.states[OnTop].get_value(support_obj):
            return True
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception("OnTop state query", exc, "Falling back to support-contact heuristics.")

    try:
        touching_support = Touching in target_obj.states and target_obj.states[Touching].get_value(support_obj)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "Touching state query",
            exc,
            "Treating this step as if no reliable contact signal is available.",
        )
        touching_support = False

    try:
        if VerticalAdjacency in target_obj.states:
            adjacency = target_obj.states[VerticalAdjacency].get_value()
            negative_neighbors = getattr(adjacency, "negative_neighbors", set())
            support_below_target = any(is_same_object(neighbor, support_obj) for neighbor in negative_neighbors)
        else:
            support_below_target = False
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "Vertical adjacency query",
            exc,
            "Disabling the support-below-target fallback for this step.",
        )
        support_below_target = False

    if not support_below_target:
        return False
    if touching_support:
        return True
    if support_surface_height is None:
        return False

    try:
        center, _, extent, _ = target_obj.get_base_aligned_bbox(xy_aligned=True)
        bottom_gap_above_support = (center[2] - extent[2] / 2.0).item() - support_surface_height
        return bottom_gap_above_support <= support_contact_tolerance
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "Target bounding-box query",
            exc,
            "Disabling the geometric support-contact fallback for this step.",
        )
        return False


def _warn_exception(context, exc, fallback_message):
    log.warning(f"[RewardSupport] {context} failed with {type(exc).__name__}. {fallback_message}")


def _iter_support_candidate_groups(task, env):
    yield task.object_scope.items()
    yield (
        (getattr(obj, "name", f"scene_obj_{idx}"), obj)
        for idx, obj in enumerate(getattr(getattr(env, "scene", None), "objects", []))
    )


def _should_skip_candidate(obj, target_obj):
    return obj is None or is_same_object(obj, target_obj) or getattr(obj, "synset", None) == "agent"


def _normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
